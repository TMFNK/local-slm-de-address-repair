import sys

sys.path.insert(0, "src")

from addr_repair.schema import validate_output


def _payload(**overrides):
    base = {
        "clean_record": {
            "name": "Example GmbH",
            "road": "Musterstraße",
            "house_number": "12",
            "postcode": "80331",
            "locality": "München",
            "country_code": "DE",
        },
        "changes": [{"field": "road", "from": "Musterstr.", "to": "Musterstraße"}],
        "needs_review": [],
    }
    base.update(overrides)
    return base


def test_schema_accepts_valid():
    ok, _ = validate_output(_payload())
    assert ok


def test_schema_rejects_missing_key():
    ok, _ = validate_output({"clean_record": {}, "changes": []})
    assert not ok
