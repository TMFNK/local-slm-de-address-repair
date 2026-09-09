import csv
import sys

import pytest

sys.path.insert(0, "src")

from addr_repair.io import entity_split, load_paired_records, records_hash


FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def _row(entity_id, road):
    return {
        "id": entity_id,
        "name": "Example GmbH",
        "road": road,
        "house_number": "12",
        "postcode": "80331",
        "locality": "München",
        "country_code": "DE",
    }


def test_entity_split_is_sorted_and_disjoint():
    result = entity_split(["e3", "e1", "e2", "e4"], 2, 1, 1)
    assert result == {"train": ["e1", "e2"], "val": ["e3"], "test": ["e4"]}


def test_entity_split_rejects_too_few_entities():
    with pytest.raises(ValueError, match="need 4 unique entities"):
        entity_split(["e1", "e2"], 2, 1, 1)


def test_load_paired_records_preserves_dirty_and_gold(tmp_path):
    dirty = tmp_path / "dirty.csv"
    clean = tmp_path / "clean.csv"
    _write_csv(dirty, [_row("e1", "Musterstr.")])
    _write_csv(clean, [_row("e1", "Musterstraße")])

    records = load_paired_records(dirty, clean)

    assert records == [
        {
            "id": "e1",
            "dirty": {field: value for field, value in _row("e1", "Musterstr.").items() if field != "id"},
            "gold": {field: value for field, value in _row("e1", "Musterstraße").items() if field != "id"},
        }
    ]


def test_load_paired_records_rejects_misaligned_ids(tmp_path):
    dirty = tmp_path / "dirty.csv"
    clean = tmp_path / "clean.csv"
    _write_csv(dirty, [_row("e1", "Musterstr.")])
    _write_csv(clean, [_row("e2", "Musterstraße")])

    with pytest.raises(ValueError, match="IDs differ"):
        load_paired_records(dirty, clean)


def test_records_hash_is_stable_for_mapping_order():
    assert records_hash([{"id": "e1", "dirty": {"road": "A"}}]) == records_hash(
        [{"dirty": {"road": "A"}, "id": "e1"}]
    )
