"""Dataset manifests, hashes, and paired-record loading. No raw data in git."""

import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def entity_split(ids: list[str], train: int, val: int, test: int, seed: int = 7) -> dict:
    """Return deterministic, disjoint entity-id partitions.

    ``seed`` is retained in the API for an explicit future policy change. The
    current policy sorts IDs, which makes a manifest independent of hash seed
    and row order.
    """
    del seed
    ordered = sorted(set(ids))
    required = train + val + test
    if len(ordered) < required:
        raise ValueError(f"need {required} unique entities, found {len(ordered)}")
    return {
        "train": ordered[:train],
        "val": ordered[train : train + val],
        "test": ordered[train + val : train + val + test],
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list")
        return payload
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _record(row: dict, row_number: int) -> dict:
    missing = [field for field in FIELDS if field not in row]
    if missing:
        raise ValueError(f"row {row_number} is missing fields: {', '.join(missing)}")
    record = {field: str(row.get(field) or "") for field in FIELDS}
    # Canonical form per the experiment contract: ISO alpha-2 uppercase.
    # The source release uses lowercase 'de'; the schema demands 'DE'.
    record["country_code"] = record["country_code"].strip().upper()
    return record


def load_paired_records(
    dirty_path: str | Path, clean_path: str | Path, id_field: str = "id"
) -> list[dict]:
    """Load aligned dirty/clean rows and return six-field paired records."""
    dirty_rows = _read_rows(Path(dirty_path))
    clean_rows = _read_rows(Path(clean_path))
    if len(dirty_rows) != len(clean_rows):
        raise ValueError("dirty and clean files have different row counts")

    records = []
    seen: set[str] = set()
    for number, (dirty_row, clean_row) in enumerate(zip(dirty_rows, clean_rows), start=2):
        dirty_id = str(dirty_row.get(id_field, number - 2))
        clean_id = str(clean_row.get(id_field, number - 2))
        if dirty_id != clean_id:
            raise ValueError(f"row {number}: dirty and clean IDs differ")
        if dirty_id in seen:
            raise ValueError(f"duplicate entity ID: {dirty_id}")
        seen.add(dirty_id)
        records.append(
            {
                "id": dirty_id,
                "dirty": _record(dirty_row, number),
                "gold": _record(clean_row, number),
            }
        )
    return records


def records_hash(records: Iterable[dict]) -> str:
    """Hash canonical record JSON, independent of input formatting."""
    encoded = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_jsonl(path: str | Path, records: Iterable[dict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return target
