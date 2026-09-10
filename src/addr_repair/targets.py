"""Evidence-preserving training targets (Fix 2, Step 5 rule).

Naive dirty-to-gold training teaches invention: where the dirty record has
an empty postcode, gold holds the geocoder answer (e.g. dirty '' -> gold
'44892'). The model cannot see the geocoder, so it learns "when in doubt,
make up a postcode".

Step 5 policy (decided 2026-09-09): training targets follow the contract,
not the raw gold. Any field empty in the dirty input keeps the empty value
in the target and is listed in needs_review, even though gold holds a value.
Fields with evidence in the input train against gold normally.

v1 policy:
- dirty empty -> keep dirty (empty), no changes entry, flag review if the
  field is in REVIEW_FIELDS (road, postcode, house_number, country_code).
  This matches rules.py after Fix 1.
- dirty non-empty -> target gold, with a changes entry when gold differs.
  v1 trusts non-empty evidence. A stricter evidence predicate (edit-distance
  allowlist, normalisation-only diffs) is a future tightening, not v1.
- changes are derived ONLY from dirty -> target diffs, never dirty -> gold.
  Invention rows therefore produce no changes entry by construction.

Known limitation (issue #8): io.py upper-cases country_code at load, so raw
dirty 'de' arrives as 'DE' and no longer forms a repair example. targets.py
works on loaded records; the raw-vs-canonical split is fixed with training
(Fix 5), not here.
"""

from __future__ import annotations

import json

from addr_repair.prompts import build_repair_prompt

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]

# Must stay in sync with rules.py empty-field review policy (Fix 1).
REVIEW_FIELDS = ("road", "postcode", "house_number", "country_code")


def _str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def build_evidence_preserving_target(
    dirty: dict, gold: dict
) -> tuple[dict, list[dict], list[str]]:
    """Return (target_clean, changes, needs_review) under the Step 5 rule."""
    target: dict = {}
    changes: list[dict] = []
    needs_review: list[str] = []

    for field in FIELDS:
        d = _str(dirty.get(field, ""))
        g = _str(gold.get(field, ""))
        if not d.strip():
            # No evidence in input: abstain, even if gold is filled.
            target[field] = d
            if field in REVIEW_FIELDS:
                needs_review.append(field)
            continue
        target[field] = g
        if g != d:
            changes.append({"field": field, "from": d, "to": g})

    return target, changes, needs_review


def build_assistant_payload(dirty: dict, gold: dict) -> dict:
    """Return the assistant JSON payload ({clean_record, changes, needs_review})."""
    target, changes, needs_review = build_evidence_preserving_target(dirty, gold)
    return {"clean_record": target, "changes": changes, "needs_review": needs_review}


def build_chat_messages(dirty: dict, gold: dict) -> list[dict]:
    """Return SFT chat messages: system + user (prompt) + assistant (target JSON)."""
    messages = build_repair_prompt(dirty)
    payload = build_assistant_payload(dirty, gold)
    messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
    return messages
