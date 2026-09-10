import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from addr_repair.schema import validate_output, validate_output_semantics
from addr_repair.targets import (
    build_assistant_payload,
    build_chat_messages,
    build_evidence_preserving_target,
)


def _dirty(**overrides):
    base = {
        "name": "Example GmbH",
        "road": "Musterstr.",
        "house_number": "12",
        "postcode": "80331",
        "locality": "München",
        "country_code": "DE",
    }
    base.update(overrides)
    return base


def test_empty_postcode_abstains_despite_gold_fill():
    # Core training-trap case: dirty '' -> gold '44892' must NOT teach '44892'.
    dirty = _dirty(postcode="")
    gold = _dirty(postcode="44892")
    target, changes, needs_review = build_evidence_preserving_target(dirty, gold)
    assert target["postcode"] == ""
    assert "postcode" in needs_review
    assert all(c["field"] != "postcode" for c in changes)
    assert "44892" not in json.dumps({"target": target, "changes": changes})


def test_empty_country_code_abstains_despite_gold_de():
    dirty = _dirty(country_code="")
    gold = _dirty(country_code="DE")
    target, changes, needs_review = build_evidence_preserving_target(dirty, gold)
    assert target["country_code"] == ""
    assert "country_code" in needs_review
    assert all(c["field"] != "country_code" for c in changes)


def test_clean_field_has_no_change_or_review():
    dirty = _dirty()
    gold = _dirty()
    target, changes, needs_review = build_evidence_preserving_target(dirty, gold)
    assert target == gold
    assert changes == []
    assert needs_review == []


def test_nonempty_repair_copies_gold_with_change_record():
    dirty = _dirty(road="Musterstr.")
    gold = _dirty(road="Musterstraße")
    target, changes, needs_review = build_evidence_preserving_target(dirty, gold)
    assert target["road"] == "Musterstraße"
    assert {"field": "road", "from": "Musterstr.", "to": "Musterstraße"} in changes
    assert needs_review == []


def test_target_payload_passes_schema_and_semantics():
    dirty = _dirty(postcode="", country_code="", road="Musterstr.")
    gold = _dirty(postcode="44892", country_code="DE", road="Musterstraße")
    payload = build_assistant_payload(dirty, gold)
    ok, _ = validate_output(payload)
    assert ok
    assert validate_output_semantics(dirty, payload) == []
    assert payload["clean_record"]["postcode"] == ""
    assert payload["clean_record"]["country_code"] == ""


def test_chat_messages_end_with_valid_assistant_json():
    dirty = _dirty(postcode="")
    gold = _dirty(postcode="44892")
    messages = build_chat_messages(dirty, gold)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[-1]["role"] == "assistant"
    payload = json.loads(messages[-1]["content"])
    assert payload["clean_record"]["postcode"] == ""
    assert "postcode" in payload["needs_review"]


def test_smoke_fixture_first_empty_postcode_row_abstains():
    fixture = Path("fixtures/smoke_100/pairs.jsonl")
    if not fixture.is_file():
        return
    rows = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
    row = next(r for r in rows if not (r["dirty"].get("postcode") or "").strip())
    target, _changes, needs_review = build_evidence_preserving_target(row["dirty"], row["gold"])
    assert target["postcode"] == ""
    assert "postcode" in needs_review
    assert row["gold"]["postcode"] != ""  # gold really was filled: the trap is real


def test_no_target_invents_on_empty_dirty_across_smoke():
    fixture = Path("fixtures/smoke_100/pairs.jsonl")
    if not fixture.is_file():
        return
    rows = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
    invented = 0
    invalid = 0
    for row in rows:
        dirty, gold = row["dirty"], row["gold"]
        payload = build_assistant_payload(dirty, gold)
        for field in ("road", "postcode", "house_number", "country_code", "name", "locality"):
            d = str(dirty.get(field) or "")
            t = str(payload["clean_record"].get(field) or "")
            if not d.strip() and t.strip():
                invented += 1
        ok, _ = validate_output(payload)
        errs = validate_output_semantics(dirty, payload)
        if not ok or errs:
            invalid += 1
    assert invented == 0
    assert invalid == 0
