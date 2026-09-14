"""Dataset manifests, hashes, and paired-record loading. No raw data in git."""

import csv
import hashlib
import json
import random
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


def pair_fingerprint(record: dict) -> str:
    """Return a canonical fingerprint for one dirty-plus-gold pair."""
    pair = {
        side: {field: str(record[side].get(field) or "") for field in FIELDS}
        for side in ("dirty", "gold")
    }
    encoded = json.dumps(pair, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def split_diversity(
    splits: dict[str, list[str]], fingerprints: dict[str, str]
) -> dict[str, int]:
    """Return the distinct content count per split for a fingerprint lookup."""
    return {
        split: len({fingerprints[entity_id] for entity_id in entity_ids})
        for split, entity_ids in splits.items()
    }


def assert_split_diversity(
    splits: dict[str, list[str]],
    fingerprints: dict[str, str],
    minimum_fraction: float = 0.10,
) -> None:
    """Refuse splits where a split repeats a few contents many times.

    A seeded shuffle almost always deals a varied hand, but a giant group can
    still land so that one split holds only a handful of distinct pairs. That
    would make training and evaluation vacuous, so fail loudly with the seed
    to change instead of shipping the manifests.
    """
    distinct = split_diversity(splits, fingerprints)
    for split, entity_ids in splits.items():
        if entity_ids and distinct[split] < minimum_fraction * len(entity_ids):
            raise ValueError(
                f"{split} holds only {distinct[split]} distinct pairs for "
                f"{len(entity_ids)} rows; change split_seed and regenerate"
            )


def _exact_subset(
    available: list[tuple[str, list[str]]], target: int, split_name: str
) -> set[int]:
    """Return group indexes whose sizes sum to exactly target.

    Reachable totals are tracked as a bit mask so the search stays fast even
    with tens of thousands of groups. Groups larger than the target can never
    fit and are skipped. Raises instead of splitting a group.
    """
    mask_limit = (1 << (target + 1)) - 1
    reachable = 1
    masks = [reachable]
    for _, ids in available:
        size = len(ids)
        if size <= target:
            reachable |= (reachable << size) & mask_limit
        masks.append(reachable)
        if (reachable >> target) & 1:
            break
    if not (reachable >> target) & 1:
        raise ValueError(
            f"cannot allocate {target} records to {split_name} without splitting "
            "a duplicate-content group"
        )
    chosen: set[int] = set()
    total = target
    for index in range(len(masks) - 1, 0, -1):
        if total == 0:
            break
        if (masks[index - 1] >> total) & 1:
            continue
        chosen.add(index - 1)
        total -= len(available[index - 1][1])
    return chosen


def pair_group_split(
    records: list[dict], train: int, val: int, test: int, seed: int = 7
) -> dict[str, list[str]]:
    """Return deterministic splits without separating duplicate content groups.

    Groups are dealt in seeded-shuffle order, so every split spans many
    different contents instead of the most-duplicated head of the data. Each
    group is assigned wholly to one split with exact row counts. The same
    seed always gives the same splits.
    """
    targets = {"train": train, "val": val, "test": test}
    if any(value < 0 for value in targets.values()):
        raise ValueError("split sizes must be non-negative")
    if len(records) < sum(targets.values()):
        raise ValueError(f"need {sum(targets.values())} records, found {len(records)}")

    groups: dict[str, list[str]] = {}
    for record in records:
        groups.setdefault(pair_fingerprint(record), []).append(record["id"])

    available = list(groups.items())
    random.Random(seed).shuffle(available)

    result: dict[str, list[str]] = {name: [] for name in targets}
    for split_name, target in targets.items():
        if target == 0:
            continue
        chosen = _exact_subset(available, target, split_name)
        for index, (_, ids) in enumerate(available):
            if index in chosen:
                result[split_name].extend(ids)
        available = [
            group for index, group in enumerate(available) if index not in chosen
        ]
    return result


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
