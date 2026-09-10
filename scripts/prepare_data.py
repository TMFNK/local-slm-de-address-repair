"""Prepare paired data: verify source, write manifests, and build smoke fixture.

Usage:
    uv run python scripts/prepare_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from addr_repair.io import (  # noqa: E402
    entity_split,
    load_paired_records,
    records_hash,
    sha256_file,
    write_jsonl,
    write_manifest,
)


def _path(raw_dir: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else raw_dir / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--force", action="store_true", help="replace generated manifests and fixture")
    parser.add_argument(
        "--skip-archive", action="store_true",
        help="skip the source-archive existence/checksum gate (fallback when only "
        "slice CSVs are present, e.g. Drive upload while Zenodo is down; "
        "dirty/clean hashes are still recorded).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_dir = Path(args.raw_dir)
    dirty_path = _path(raw_dir, config.get("dirty_file", "dirty.csv"))
    clean_path = _path(raw_dir, config.get("clean_file", "clean.csv"))
    archive_path = _path(raw_dir, config["archive_file"]) if config.get("archive_file") else None
    if args.skip_archive:
        archive_path = None
    if archive_path and not archive_path.is_file():
        raise SystemExit(f"Source archive is not available: {archive_path}")
    expected_archive_hash = config.get("archive_sha256", "")
    if archive_path and expected_archive_hash and not expected_archive_hash.startswith("REPLACE_"):
        actual_archive_hash = sha256_file(archive_path)
        if actual_archive_hash != expected_archive_hash:
            raise SystemExit(
                f"Archive checksum mismatch: expected {expected_archive_hash}, "
                f"got {actual_archive_hash}"
            )
    missing = [str(path) for path in (dirty_path, clean_path) if not path.is_file()]
    if missing:
        raise SystemExit(
            "Source data is not available. Download the pinned release first; missing: "
            + ", ".join(missing)
        )

    records = load_paired_records(dirty_path, clean_path, config.get("id_field", "id"))
    split_config = config["splits"]
    splits = entity_split(
        [record["id"] for record in records],
        split_config["train_entities"],
        split_config["val_entities"],
        split_config["test_entities"],
    )
    by_id = {record["id"]: record for record in records}
    if len(by_id) != len(records):
        raise SystemExit("Input contains duplicate entity IDs")

    manifests_dir = Path(config.get("manifests_dir", "data/manifests"))
    if manifests_dir.exists() and any(manifests_dir.iterdir()) and not args.force:
        raise SystemExit(f"{manifests_dir} is not empty; use --force to replace generated files")
    fixture_dir = Path(config.get("smoke_fixture", "fixtures/smoke_100"))
    if fixture_dir.exists() and any(fixture_dir.iterdir()) and not args.force:
        raise SystemExit(f"{fixture_dir} is not empty; use --force to replace generated files")

    generated_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    source = {
        "dirty_file": str(dirty_path),
        "clean_file": str(clean_path),
        "dirty_sha256": sha256_file(dirty_path),
        "clean_sha256": sha256_file(clean_path),
        "archive": str(archive_path) if archive_path else None,
        "archive_sha256": sha256_file(archive_path) if archive_path else None,
        "archive_skipped": bool(args.skip_archive),
        "source_repo": config["source_repo"],
        "zenodo_url": config["zenodo_url"],
        "dataset_date": config["dataset_date"],
        "license": config["license"],
        "generated_at": generated_at,
    }
    for split_name, ids in splits.items():
        split_records = [by_id[entity_id] for entity_id in ids]
        write_manifest(
            manifests_dir / f"{split_name}.json",
            {
                **source,
                "split": split_name,
                "entity_count": len(ids),
                "entity_ids": ids,
                "records_sha256": records_hash(split_records),
                "test_is_original_pairs": split_name == "test",
            },
        )

    smoke_records = [by_id[entity_id] for entity_id in splits["test"][:100]]
    write_jsonl(fixture_dir / "pairs.jsonl", smoke_records)
    write_manifest(
        fixture_dir / "manifest.json",
        {
            **source,
            "split": "test",
            "entity_count": len(smoke_records),
            "entity_ids": [record["id"] for record in smoke_records],
            "records_sha256": records_hash(smoke_records),
            "test_is_original_pairs": True,
        },
    )
    print(f"[prepare_data] wrote {len(records)} paired records")
    print(f"[prepare_data] splits={{{', '.join(f'{name}: {len(ids)}' for name, ids in splits.items())}}}")
    print(f"[prepare_data] smoke={len(smoke_records)} records at {fixture_dir}")


if __name__ == "__main__":
    main()
