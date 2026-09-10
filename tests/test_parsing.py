import json
import sys

sys.path.insert(0, "src")

from addr_repair.parsing import check_contract, extract_json, parse_response


def test_extract_json_plain_object():
    parsed, error = extract_json('{"a": 1}')
    assert error is None
    assert parsed == {"a": 1}


def test_extract_json_strips_fences():
    parsed, error = extract_json('```json\n{"clean_record": {}}\n```')
    assert error is None
    assert parsed == {"clean_record": {}}


def test_extract_json_finds_embedded_object():
    parsed, error = extract_json('Here is the repair:\n{"a": 1}\nDone.')
    assert error is None
    assert parsed == {"a": 1}


def test_extract_json_rejects_prose():
    parsed, error = extract_json("no json here")
    assert parsed is None
    assert error == "no-json-object-found"


def test_check_contract_rejects_flat_clean_record():
    errors = check_contract(
        {"clean_record": "name: X, road: Y", "changes": [], "needs_review": []}
    )
    assert errors == ["clean_record-is-not-an-object"]


def test_check_contract_rejects_missing_keys():
    errors = check_contract({"clean_record": {}})
    assert "missing-key: changes" in errors
    assert "missing-key: needs_review" in errors
    assert any(e.startswith("clean_record-missing-field:") for e in errors)


def test_parse_response_accepts_valid_output():
    record = {f: "" for f in ("name", "road", "house_number", "postcode", "locality", "country_code")}
    parsed, errors = parse_response(
        json.dumps(
            {"clean_record": record, "changes": [], "needs_review": ["postcode"]}
        )
    )
    assert errors == []
    assert parsed["needs_review"] == ["postcode"]
