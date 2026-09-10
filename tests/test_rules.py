import sys

sys.path.insert(0, "src")

from addr_repair.rules import repair_with_rules


def test_rules_fix_whitespace_and_case():
    repaired, changes, _ = repair_with_rules(
        {
            "name": "  Example GmbH ",
            "road": "Musterstr.",
            "house_number": "12",
            "postcode": "80331",
            "locality": "München",
            "country_code": "de",
        }
    )
    assert repaired["name"] == "Example GmbH"
    assert repaired["country_code"] == "DE"
    assert any(c["field"] == "name" for c in changes)


def test_rules_never_invents_missing_road():
    repaired, _, needs_review = repair_with_rules(
        {
            "name": "Example GmbH",
            "road": "",
            "house_number": "",
            "postcode": "80331",
            "locality": "München",
            "country_code": "DE",
        }
    )
    assert repaired["road"] == ""
    assert "road" in needs_review


def test_rules_expands_glued_str_suffix():
    repaired, changes, _ = repair_with_rules(
        {
            "name": "Example GmbH",
            "road": "Musterstr.",
            "house_number": "12",
            "postcode": "80331",
            "locality": "München",
            "country_code": "DE",
        }
    )
    assert repaired["road"] == "Musterstraße"
    assert {"field": "road", "from": "Musterstr.", "to": "Musterstraße"} in changes


def test_rules_flags_empty_country_code_for_review():
    # Fix 1: empty country_code is abstention, not a schema violation.
    repaired, changes, needs_review = repair_with_rules(
        {
            "name": "Example GmbH",
            "road": "Musterstraße",
            "house_number": "12",
            "postcode": "80331",
            "locality": "München",
            "country_code": "",
        }
    )
    assert repaired["country_code"] == ""
    assert "country_code" in needs_review
    assert all(c["field"] != "country_code" for c in changes)
