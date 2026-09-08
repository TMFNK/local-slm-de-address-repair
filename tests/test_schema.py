import sys

sys.path.insert(0, "src")

from addr_repair.schema import validate_output, validate_output_semantics


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


def test_semantics_accept_matching_change_record():
    dirty = _payload()["clean_record"] | {"road": "Musterstr."}
    payload = _payload()
    assert validate_output_semantics(dirty, payload) == []


def test_semantics_reject_unreported_clean_record_change():
    dirty = _payload()["clean_record"]
    payload = _payload(
        clean_record={**dirty, "road": "Falschweg"},
        changes=[],
    )
    errors = validate_output_semantics(dirty, payload)
    assert "clean_record[road] changed without a change record" in errors


def test_semantics_reject_change_and_review_for_same_field():
    dirty = _payload()["clean_record"] | {"road": "Musterstr."}
    payload = _payload(needs_review=["road"])
    errors = validate_output_semantics(dirty, payload)
    assert "road appears in both changes and needs_review" in errors
