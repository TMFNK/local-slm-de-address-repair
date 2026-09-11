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
    assert metrics["review_field_rate"] == 0.0
    assert metrics["review_record_coverage"] == 0.0
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
    assert metrics["review_field_rate"] == round(1 / 6, 4)
    assert metrics["review_record_coverage"] == 1.0


def test_scorer_does_not_reward_invented_correct_gold():
    # Fix 3: dirty '' -> gold '44892', pred '44892' without review is
    # invention, not a correct repair — even though it matches gold.
    dirty = {
        "name": "x",
        "road": "Musterstraße",
        "house_number": "12",
        "postcode": "",
        "locality": "München",
        "country_code": "DE",
    }
    gold = {**dirty, "postcode": "44892"}
    pred = {**dirty, "postcode": "44892"}
    metrics = score_records(
        [
            {
                "dirty": dirty,
                "gold": gold,
                "pred": pred,
                "needs_review": [],
                "schema_ok": True,
                "semantic_ok": True,
                "parsed": True,
            }
        ]
    )
    assert metrics["by_field"]["postcode"]["correct_repairs"] == 0
    assert metrics["by_field"]["postcode"]["inventions"] == 1
    assert metrics["inventions"] == 1
    assert metrics["repair_precision"] == 0.0
    # The invented field is a miss against honest behaviour: no TP earned.
    assert metrics["repair_recall"] == 0.0


def test_scorer_counts_invention_even_with_review_flag():
    # A review flag does not excuse filling: semantic layer rejects it
    # separately, and the scorer still counts invention (never TP).
    dirty = {
        "name": "x",
        "road": "Musterstraße",
        "house_number": "12",
        "postcode": "",
        "locality": "München",
        "country_code": "DE",
    }
    gold = {**dirty, "postcode": "44892"}
    pred = {**dirty, "postcode": "44892"}
    metrics = score_records(
        [
            {
                "dirty": dirty,
                "gold": gold,
                "pred": pred,
                "needs_review": ["postcode"],
                "schema_ok": True,
                "semantic_ok": False,
                "parsed": True,
            }
        ]
    )
    assert metrics["by_field"]["postcode"]["correct_repairs"] == 0
    assert metrics["inventions"] == 1
    assert metrics["semantic_validity"] == 0.0
    assert metrics["contract_validity"] == 0.0


def test_scorer_counts_wrong_invention_once():
    # Invented but wrong: one FP (not double), plus the recall miss.
    dirty = {
        "name": "x",
        "road": "Musterstraße",
        "house_number": "12",
        "postcode": "",
        "locality": "München",
        "country_code": "DE",
    }
    gold = {**dirty, "postcode": "44892"}
    pred = {**dirty, "postcode": "99999"}
    metrics = score_records(
        [
            {
                "dirty": dirty,
                "gold": gold,
                "pred": pred,
                "needs_review": [],
                "schema_ok": True,
                "semantic_ok": True,
                "parsed": True,
            }
        ]
    )
    assert metrics["inventions"] == 1
    assert metrics["repair_precision"] == 0.0
    assert metrics["repair_recall"] == 0.0
    assert metrics["by_field"]["postcode"]["inventions"] == 1


def test_scorer_counts_wrong_nonempty_addition_separately():
    dirty = {
        "name": "Pfarrhaus",
        "road": "Heinrich-Schliemann-Straße",
        "house_number": "4",
        "postcode": "23942",
        "locality": "Kalkhorst",
        "country_code": "DE",
    }
    metrics = score_records(
        [
            {
                "dirty": dirty,
                "gold": dict(dirty),
                "pred": {**dirty, "name": "Pfarrhaus Kalkhorst"},
                "needs_review": [],
                "schema_ok": True,
                "semantic_ok": True,
                "parsed": True,
            }
        ]
    )
    assert metrics["inventions"] == 0
    assert metrics["unsupported_additions"] == 1
    assert metrics["unsupported_addition_rate"] == round(1 / 6, 4)
    assert metrics["by_field"]["name"]["unsupported_additions"] == 1


def test_scorer_abstention_still_counts_as_recall_miss():
    # Step 5 contract: honest abstention is correct behaviour but still a
    # recall miss on the gold-based test set. Locks the reported tension.
    dirty = {
        "name": "x",
        "road": "Musterstraße",
        "house_number": "12",
        "postcode": "",
        "locality": "München",
        "country_code": "DE",
    }
    gold = {**dirty, "postcode": "44892"}
    metrics = score_records(
        [
            {
                "dirty": dirty,
                "gold": gold,
                "pred": dict(dirty),
                "needs_review": ["postcode"],
                "schema_ok": True,
                "semantic_ok": True,
                "parsed": True,
            }
        ]
    )
    assert metrics["inventions"] == 0
    assert metrics["by_field"]["postcode"]["correct_abstentions"] == 1
    assert metrics["repair_recall"] == 0.0
    assert metrics["review_precision"] == 1.0


def test_scorer_reports_validity_split_and_contract():
    base = {
        "name": "x",
        "road": "Musterstraße",
        "house_number": "12",
        "postcode": "80331",
        "locality": "München",
        "country_code": "DE",
    }
    rows = [
        {
            "dirty": dict(base),
            "gold": dict(base),
            "pred": dict(base),
            "needs_review": [],
            "schema_ok": True,
            "semantic_ok": True,
            "parsed": True,
        },
        {
            "dirty": dict(base),
            "gold": dict(base),
            "pred": dict(base),
            "needs_review": [],
            "schema_ok": True,
            "semantic_ok": False,
            "parsed": True,
        },
        {
            "dirty": dict(base),
            "gold": dict(base),
            "pred": dict(base),
            "needs_review": [],
            "schema_ok": False,
            "semantic_ok": False,
            "parsed": False,
        },
    ]
    metrics = score_records(rows)
    assert metrics["schema_validity"] == round(2 / 3, 4)
    assert metrics["semantic_validity"] == round(1 / 3, 4)
    assert metrics["parse_rate"] == round(2 / 3, 4)
    assert metrics["contract_validity"] == round(1 / 3, 4)
    assert metrics["inventions"] == 0
    assert metrics["invention_rate"] == 0.0
