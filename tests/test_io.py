import csv
import sys

import pytest

sys.path.insert(0, "src")

from addr_repair.io import (
    assert_split_diversity,
    load_paired_records,
    pair_fingerprint,
    pair_group_split,
    records_hash,
)

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


def test_pair_group_split_keeps_duplicate_pairs_together():
    def pair(entity_id, dirty_road, gold_road):
        return {
            "id": entity_id,
            "dirty": {**_row(entity_id, dirty_road), "id": None},
            "gold": {**_row(entity_id, gold_road), "id": None},
        }

    records = [
        pair("e1", "Musterstr.", "Musterstraße"),
        pair("e2", "Musterstr.", "Musterstraße"),
        pair("e3", "Hauptstr.", "Hauptstraße"),
        pair("e4", "Bahnhofstr.", "Bahnhofstraße"),
    ]

    result = pair_group_split(records, 2, 1, 1)
    assignments = {
        entity_id: split
        for split, entity_ids in result.items()
        for entity_id in entity_ids
    }

    assert {len(entity_ids) for entity_ids in result.values()} == {1, 2}
    assert assignments["e1"] == assignments["e2"]
    assert set(assignments) == {"e1", "e2", "e3", "e4"}


def test_pair_group_split_rejects_unfillable_group_size():
    records = [
        {
            "id": f"e{i}",
            "dirty": _row(f"e{i}", "Musterstr."),
            "gold": _row(f"e{i}", "Musterstraße"),
        }
        for i in range(2)
    ]

    with pytest.raises(ValueError, match="without splitting"):
        pair_group_split(records, 1, 1, 0)


def test_pair_group_split_is_deterministic_for_a_seed():
    def pair(entity_id, dirty_road, gold_road):
        return {
            "id": entity_id,
            "dirty": {**_row(entity_id, dirty_road), "id": None},
            "gold": {**_row(entity_id, gold_road), "id": None},
        }

    records = [pair(f"e{i}", f"Straße {i}", f"Strasse {i}") for i in range(20)]
    first = pair_group_split(records, 10, 5, 5, seed=7)
    second = pair_group_split(records, 10, 5, 5, seed=7)

    assert first == second


def test_pair_group_split_spreads_duplicate_head_across_splits():
    def pair(entity_id, dirty_road, gold_road):
        return {
            "id": entity_id,
            "dirty": {**_row(entity_id, dirty_road), "id": None},
            "gold": {**_row(entity_id, gold_road), "id": None},
        }

    head = [pair("dup0", "Musterstr.", "Musterstraße")]
    tail = [pair(f"u{i}", f"Straße {i}", f"Strasse {i}") for i in range(30)]
    records = head + tail

    result = pair_group_split(records, 10, 5, 5, seed=7)
    by_id = {record["id"]: record for record in records}
    distinct_per_split = {
        split: {pair_fingerprint(by_id[entity_id]) for entity_id in entity_ids}
        for split, entity_ids in result.items()
    }

    assert {len(ids) for ids in result.values()} == {5, 10}
    train_fps, val_fps, test_fps = (
        distinct_per_split["train"],
        distinct_per_split["val"],
        distinct_per_split["test"],
    )
    assert not (train_fps & val_fps)
    assert not (train_fps & test_fps)
    assert not (val_fps & test_fps)
    assert_split_diversity(
        result, {record["id"]: pair_fingerprint(record) for record in records}
    )


def test_assert_split_diversity_refuses_repeated_contents():
    splits = {"train": [f"e{i}" for i in range(12)]}
    fingerprints = {f"e{i}": "same" for i in range(12)}

    with pytest.raises(ValueError, match="change split_seed"):
        assert_split_diversity(splits, fingerprints)

    healthy = {"train": ["a", "b", "c", "d"]}
    healthy_fps = {"a": "w", "b": "x", "c": "y", "d": "z"}
    assert_split_diversity(healthy, healthy_fps)


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


def test_country_code_is_canonicalized_to_uppercase(tmp_path):
    dirty = tmp_path / "dirty.csv"
    clean = tmp_path / "clean.csv"
    dirty_row = _row("e1", "Musterstr.")
    dirty_row["country_code"] = "de"
    clean_row = _row("e1", "Musterstraße")
    clean_row["country_code"] = "de"
    _write_csv(dirty, [dirty_row])
    _write_csv(clean, [clean_row])

    (record,) = load_paired_records(dirty, clean)

    assert record["dirty"]["country_code"] == "DE"
    assert record["gold"]["country_code"] == "DE"
