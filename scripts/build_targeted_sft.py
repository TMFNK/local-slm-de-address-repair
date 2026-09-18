"""Build an audited, train-only manifest for targeted SFT.

The targeted view contains every pinned-train pair where ``name`` or
``locality`` differs, plus every all-clean train pair as a damage-control
set. Validation and test manifests are copied unchanged.

Usage:
    uv run python scripts/build_targeted_sft.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from addr_repair.io import load_paired_records, records_hash, sha256_file
from addr_repair.scorer import FIELDS

TARGET_FIELDS = ("name", "locality")


def _path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_split(
    manifest_dir: Path,
    raw_dir: Path,
    split: str,
    dirty_file: str,
    clean_file: str,
    id_field: str,
) -> tuple[list[dict], dict]:
    manifest_path = manifest_dir / f"{split}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = load_paired_records(raw_dir / dirty_file, raw_dir / clean_file, id_field)
    by_id = {record["id"]: record for record in records}
    missing = [entity_id for entity_id in manifest["entity_ids"] if entity_id not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} manifest IDs missing from raw data ({split})")
    pairs = [by_id[entity_id] for entity_id in manifest["entity_ids"]]
    if records_hash(pairs) != manifest["records_sha256"]:
        raise SystemExit(f"{split} records hash differs from manifest; inputs changed")
    return pairs, manifest


def _changed_fields(pair: dict) -> set[str]:
    return {
        field
        for field in FIELDS
        if str(pair["dirty"].get(field) or "") != str(pair["gold"].get(field) or "")
    }


def _write_manifest(
    path: Path,
    pairs: list[dict],
    dirty_sha256: str,
    clean_sha256: str,
    **metadata,
) -> dict:
    payload = {
        "dirty_file": "data/raw/dirty.csv",
        "clean_file": "data/raw/clean.csv",
        "dirty_sha256": dirty_sha256,
        "clean_sha256": clean_sha256,
        "entity_ids": [pair["id"] for pair in pairs],
        "records_sha256": records_hash(pairs),
        **metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def build_targeted_view(config_path: str | Path, out_dir: str | Path) -> dict:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path.cwd()
    raw_dir = _path(config.get("raw_dir", "data/raw"), root)
    manifest_dir = _path(config.get("manifests_dir", "data/manifests"), root)
    target_dir = _path(out_dir, root)
    dirty_file = config.get("dirty_file", "dirty.csv")
    clean_file = config.get("clean_file", "clean.csv")
    id_field = config.get("id_field", "id")

    train, train_manifest = _load_split(
        manifest_dir, raw_dir, "train", dirty_file, clean_file, id_field
    )
    _val, val_manifest = _load_split(manifest_dir, raw_dir, "val", dirty_file, clean_file, id_field)
    test, test_manifest = _load_split(
        manifest_dir, raw_dir, "test", dirty_file, clean_file, id_field
    )

    targeted = [pair for pair in train if _changed_fields(pair) & set(TARGET_FIELDS)]
    controls = [pair for pair in train if not _changed_fields(pair)]
    selected = targeted + controls
    selected_ids = {pair["id"] for pair in selected}
    test_ids = {pair["id"] for pair in test}
    overlap = sorted(selected_ids & test_ids)
    if overlap:
        raise SystemExit(
            f"{len(overlap)} targeted train ids overlap test.json; refusing to write view"
        )

    dirty_sha256 = sha256_file(raw_dir / dirty_file)
    clean_sha256 = sha256_file(raw_dir / clean_file)
    target_dir.mkdir(parents=True, exist_ok=True)
    train_out = _write_manifest(
        target_dir / "train.json",
        selected,
        dirty_sha256,
        clean_sha256,
        split="train",
        source_split="train",
        source_manifest_records_sha256=train_manifest["records_sha256"],
        selection={
            "target_fields": list(TARGET_FIELDS),
            "target_rule": "name or locality differs between dirty and gold",
            "target_rows": len(targeted),
            "control_rule": "all fields equal between dirty and gold",
            "control_rows": len(controls),
        },
    )
    shutil.copyfile(manifest_dir / "val.json", target_dir / "val.json")
    shutil.copyfile(manifest_dir / "test.json", target_dir / "test.json")

    return {
        "out_dir": str(target_dir),
        "train_rows": len(selected),
        "target_rows": len(targeted),
        "control_rows": len(controls),
        "val_manifest_records_sha256": val_manifest["records_sha256"],
        "test_manifest_records_sha256": test_manifest["records_sha256"],
        "targeted_train_manifest_records_sha256": train_out["records_sha256"],
        "test_overlap": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--out-dir",
        default="data/manifests/targeted-name-locality-v1",
    )
    args = parser.parse_args()
    print(json.dumps(build_targeted_view(args.config, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
