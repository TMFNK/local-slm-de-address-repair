import sys

sys.path.insert(0, "src")

from addr_repair.scorer import score_records


def test_scorer_counts_repair_and_damage():
    rows = [
        {
            "dirty": {"name": "x", "road": "Musterstr.", "house_number": "12", "postcode": "80331", "locality": "München", "country_code": "DE"},
            "gold": {"name": "x", "road": "Musterstraße", "house_number": "12", "postcode": "80331", "locality": "München", "country_code": "DE"},
            "pred": {"name": "x", "road": "Musterstraße", "house_number": "12", "postcode": "80331", "locality": "München", "country_code": "DE"},
            "needs_review": [],
            "schema_ok": True,
        },
        {
            "dirty": {"name": "x", "road": "Musterstraße", "house_number": "12", "postcode": "80331", "locality": "München", "country_code": "DE"},
            "gold": {"name": "x", "road": "Musterstraße", "house_number": "12", "postcode": "80331", "locality": "München", "country_code": "DE"},
            "pred": {"name": "x", "road": "Falschweg", "house_number": "12", "postcode": "80331", "locality": "München", "country_code": "DE"},
            "needs_review": [],
            "schema_ok": True,
        },
    ]
    metrics = score_records(rows)
    assert metrics["n"] == 2
    assert metrics["repair_recall"] == 1.0
    assert metrics["damage_rate"] > 0.0
    assert metrics["schema_validity"] == 1.0
    assert metrics["by_field"]["road"]["correct_repairs"] == 1
    assert metrics["by_field"]["road"]["repair_recall"] == 1.0


def test_scorer_counts_abstention_and_review_for_missing_input():
    dirty = {
        "name": "x",
        "road": "",
        "house_number": "12",
        "postcode": "80331",
        "locality": "München",
        "country_code": "DE",
    }
    gold = {**dirty, "road": "Musterstraße"}
    metrics = score_records(
        [
            {
                "dirty": dirty,
                "gold": gold,
                "pred": dirty,
                "needs_review": ["road"],
                "schema_ok": True,
            }
        ]
    )
    assert metrics["by_field"]["road"]["correct_abstentions"] == 1
    assert metrics["by_field"]["road"]["review_precision"] == 1.0
    assert metrics["review_precision"] == 1.0
