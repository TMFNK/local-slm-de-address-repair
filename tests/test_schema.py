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


def test_schema_accepts_empty_country_code_abstention():
    # Fix 1: "" is a valid abstention value for country_code.
    base = _payload()["clean_record"]
    dirty = {**base, "road": "Musterstr.", "country_code": ""}
    payload = _payload(
        clean_record={**base, "road": "Musterstraße", "country_code": ""},
        changes=[{"field": "road", "from": "Musterstr.", "to": "Musterstraße"}],
        needs_review=["country_code"],
    )
    ok, _ = validate_output(payload)
    assert ok
    assert validate_output_semantics(dirty, payload) == []


def test_rules_empty_country_code_passes_full_contract():
    # End-to-end: rules -> validate_output -> semantics for empty country.
    import sys

    sys.path.insert(0, "src")
    from addr_repair.rules import repair_with_rules

    dirty = {
        "name": "Example GmbH",
        "road": "Musterstraße",
        "house_number": "12",
        "postcode": "80331",
        "locality": "München",
        "country_code": "",
    }
    repaired, changes, needs_review = repair_with_rules(dirty)
    payload = {"clean_record": repaired, "changes": changes, "needs_review": needs_review}
    ok, _ = validate_output(payload)
    assert ok
    assert validate_output_semantics(dirty, payload) == []
    assert "country_code" in needs_review


def test_semantics_reject_invention_with_change_record():
    # Fix 4: dirty '' -> '44892' with a matching change record is still
    # invention. The change bookkeeping is consistent but the value has no
    # evidence in the input.
    base = _payload()["clean_record"]
    dirty = {**base, "postcode": ""}
    payload = _payload(
        clean_record={**base, "postcode": "44892"},
        changes=[{"field": "postcode", "from": "", "to": "44892"}],
        needs_review=[],
    )
    ok, _ = validate_output(payload)
    assert ok  # shape is fine; the problem is semantic
    errors = validate_output_semantics(dirty, payload)
    assert "clean_record[postcode] invents a value for an empty dirty field" in errors


def test_semantics_reject_invention_without_change_record():
    base = _payload()["clean_record"]
    dirty = {**base, "road": ""}
    payload = _payload(
        clean_record={**base, "road": "Musterstraße"},
        changes=[],
        needs_review=[],
    )
    errors = validate_output_semantics(dirty, payload)
    assert "clean_record[road] invents a value for an empty dirty field" in errors


def test_semantics_reject_invention_even_with_review_flag():
    # Filling + flagging is doubly wrong: the flag must keep the dirty value
    # (existing check) and the fill is invention (new check).
    base = _payload()["clean_record"]
    dirty = {**base, "postcode": ""}
    payload = _payload(
        clean_record={**base, "postcode": "44892"},
        changes=[],
        needs_review=["postcode"],
    )
    errors = validate_output_semantics(dirty, payload)
    assert "clean_record[postcode] invents a value for an empty dirty field" in errors
    assert "needs_review[postcode] changes the value being reviewed" in errors


def test_semantics_require_review_for_empty_critical_field():
    # Kept empty but unrouted: the model abstained silently instead of asking.
    base = _payload()["clean_record"]
    dirty = {**base, "country_code": ""}
    payload = _payload(
        clean_record={**base, "country_code": ""},
        changes=[{"field": "road", "from": "Musterstr.", "to": "Musterstraße"}],
        needs_review=[],
    )
    dirty = {**dirty, "road": "Musterstr."}
    errors = validate_output_semantics(dirty, payload)
    assert "country_code is empty in dirty input but missing from needs_review" in errors


def test_semantics_no_review_required_for_empty_noncritical_field():
    # name/locality empties follow rules v1: kept empty, no flag needed.
    base = _payload()["clean_record"]
    dirty = {**base, "road": "Musterstr.", "name": ""}
    payload = _payload(
        clean_record={**base, "road": "Musterstraße", "name": ""},
        changes=[{"field": "road", "from": "Musterstr.", "to": "Musterstraße"}],
        needs_review=[],
    )
    assert validate_output_semantics(dirty, payload) == []


def test_semantics_whitespace_only_dirty_counts_as_empty():
    base = _payload()["clean_record"]
    dirty = {**base, "postcode": "   "}
    payload = _payload(
        clean_record={**base, "postcode": "44892"},
        changes=[{"field": "postcode", "from": "   ", "to": "44892"}],
        needs_review=[],
    )
    errors = validate_output_semantics(dirty, payload)
    assert "clean_record[postcode] invents a value for an empty dirty field" in errors


def test_semantics_accept_full_abstention_record():
    # End-to-end shape: every empty critical field kept empty + flagged.
    from addr_repair.rules import repair_with_rules

    dirty = {
        "name": "Example GmbH",
        "road": "",
        "house_number": "",
        "postcode": "",
        "locality": "München",
        "country_code": "",
    }
    repaired, changes, needs_review = repair_with_rules(dirty)
    payload = {"clean_record": repaired, "changes": changes, "needs_review": needs_review}
    ok, _ = validate_output(payload)
    assert ok
    assert validate_output_semantics(dirty, payload) == []
    assert set(["road", "house_number", "postcode", "country_code"]) <= set(needs_review)
