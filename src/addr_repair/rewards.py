"""GRPO reward for address repair (plan 2026-09-15, step 2).

Scores one model answer in [0, 1] as a sum of small parts. No training
code lives here; ``scripts/train_grpo.py`` (step 5) wraps
:func:`reward_for_training` for TRL.

Start weights (from the plan):

- parses as JSON: +0.10 / 0
- passes JSON shape check: +0.10 / 0
- passes meaning checks: +0.20 / 0
- per-field repair: +0.10 per correct fix on a non-empty dirty field,
  -0.15 per wrong change to a clean field, -0.20 per fill of an empty
  field (even when the fill matches gold)
- honest review: +0.05 per empty critical field kept empty and flagged
- total clipped to [0, 1]

Two decisions locked with the user before writing (2026-09-16):

- unsupported addition (non-empty dirty value kept as a contiguous token
  sequence plus extra tokens, e.g. ``Helmholtz-Gymnasium`` ->
  ``Helmholtz-Gymnasium Karlsruhe``): no repair credit, -0.15 each, same
  size as the clean-field damage penalty. This reuses the scorer's
  token-containment test. Edge case: if gold genuinely extends dirty the
  penalty still applies; such rows are rare in this dataset and the
  scorer reports them as additions too.
- review bonus only when the flagged field is actually empty in dirty,
  kept empty in the prediction, and listed in ``needs_review``. Flagging
  a non-empty or changed field earns nothing (reported as spurious so
  step 4 can check for flag-everything gaming).

Faithful-to-plan notes:

- a wrong edit on an already-dirty field (repaired, ``p != gold``,
  not an invention or addition) earns 0 and no penalty. It is counted
  as ``wrong_dirty`` in the breakdown so step 4 can see how often the
  +0.40 structure part outweighs a missed repair.
- ``structure_total`` (parse + shape + meaning, max 0.40) is returned
  alongside ``total`` so step 4 can rank answers by shape-only versus
  the full reward without a second code path.
- meaning reuses :func:`schema.validate_output_semantics`, which already
  rejects invented fills, mismatched ``changes``, and dirty ``needs_review``.
"""

from __future__ import annotations

from addr_repair.parsing import extract_json
from addr_repair.schema import ABSTENTION_FIELDS, validate_output, validate_output_semantics
from addr_repair.scorer import FIELDS as SCORER_FIELDS
from addr_repair.scorer import _is_unsupported_addition, _norm

FIELDS = list(SCORER_FIELDS)
CRITICAL_FIELDS = tuple(ABSTENTION_FIELDS)

PARSE_W = 0.10
SHAPE_W = 0.10
MEANING_W = 0.20
CORRECT_W = 0.10
CLEAN_DAMAGE_W = -0.15
INVENTION_W = -0.20
ADDITION_W = -0.15
HONEST_REVIEW_W = 0.05

WEIGHTS = {
    "parse": PARSE_W,
    "shape": SHAPE_W,
    "meaning": MEANING_W,
    "correct_repair": CORRECT_W,
    "clean_damage": CLEAN_DAMAGE_W,
    "invention": INVENTION_W,
    "addition": ADDITION_W,
    "honest_review": HONEST_REVIEW_W,
}


def _pred_record(payload: dict) -> dict:
    record = payload.get("clean_record")
    return record if isinstance(record, dict) else {}


def score_parsed(dirty: dict, gold: dict | None, payload: dict) -> dict:
    """Score an already-parsed answer payload.

    ``gold`` may be None (repair parts are then 0; structure and review
    parts still apply). Assumes the payload parsed, so the parse part
    is credited; use :func:`score_answer` for raw model text.
    """
    shape_ok, _ = validate_output(payload)
    semantic_errors = validate_output_semantics(dirty, payload)
    meaning_ok = bool(shape_ok) and not semantic_errors

    pred = _pred_record(payload)
    reviews = payload.get("needs_review")
    review_list = list(reviews) if isinstance(reviews, list) else []

    n_correct = n_clean_damage = n_invention = n_addition = n_wrong_dirty = 0
    if gold is not None:
        for field in FIELDS:
            d = _norm(dirty.get(field))
            g = _norm(gold.get(field))
            p = _norm(pred.get(field))
            d_empty = not d
            was_dirty = d != g
            repaired = p != d
            if not repaired:
                continue
            if d_empty:
                # Fill with no evidence: never a repair, even if p == g.
                n_invention += 1
            elif _is_unsupported_addition(d, p):
                n_addition += 1
            elif not was_dirty:
                n_clean_damage += 1
            elif p == g:
                n_correct += 1
            else:
                n_wrong_dirty += 1

    honest_fields: list[str] = []
    spurious_flags: list[str] = []
    for field in review_list:
        if field not in FIELDS:
            spurious_flags.append(field)
            continue
        d = _norm(dirty.get(field))
        p = _norm(pred.get(field))
        if not d and not p and field in CRITICAL_FIELDS:
            honest_fields.append(field)
        else:
            spurious_flags.append(field)

    parse_part = PARSE_W
    shape_part = SHAPE_W if shape_ok else 0.0
    meaning_part = MEANING_W if meaning_ok else 0.0
    repair_total = (
        CORRECT_W * n_correct
        + CLEAN_DAMAGE_W * n_clean_damage
        + INVENTION_W * n_invention
        + ADDITION_W * n_addition
    )
    review_bonus = HONEST_REVIEW_W * len(honest_fields)
    total = parse_part + shape_part + meaning_part + repair_total + review_bonus
    total = max(0.0, min(1.0, total))

    return {
        "total": round(total, 4),
        "structure_total": round(parse_part + shape_part + meaning_part, 4),
        "parse": round(parse_part, 4),
        "shape": round(shape_part, 4),
        "meaning": round(meaning_part, 4),
        "repair_total": round(repair_total, 4),
        "n_correct": n_correct,
        "n_clean_damage": n_clean_damage,
        "n_invention": n_invention,
        "n_addition": n_addition,
        "n_wrong_dirty": n_wrong_dirty,
        "review_bonus": round(review_bonus, 4),
        "honest_review_fields": honest_fields,
        "spurious_flags": spurious_flags,
        "parsed_ok": True,
        "shape_ok": bool(shape_ok),
        "meaning_ok": bool(meaning_ok),
        "semantic_errors": list(semantic_errors),
        "weights": dict(WEIGHTS),
    }


def score_answer(dirty: dict, gold: dict | None, answer_text: str) -> dict:
    """Score one raw model answer string. Unparseable text scores 0 total."""
    payload, error = extract_json(answer_text or "")
    if error is not None or payload is None:
        return {
            "total": 0.0,
            "structure_total": 0.0,
            "parse": 0.0,
            "shape": 0.0,
            "meaning": 0.0,
            "repair_total": 0.0,
            "n_correct": 0,
            "n_clean_damage": 0,
            "n_invention": 0,
            "n_addition": 0,
            "n_wrong_dirty": 0,
            "review_bonus": 0.0,
            "honest_review_fields": [],
            "spurious_flags": [],
            "parsed_ok": False,
            "shape_ok": False,
            "meaning_ok": False,
            "semantic_errors": [error or "no-json-object-found"],
            "weights": dict(WEIGHTS),
        }
    return score_parsed(dirty, gold, payload)


def reward_for_training(dirty: dict, gold: dict | None, answer_text: str) -> float:
    """Scalar reward in [0, 1] for the TRL GRPO loop."""
    return float(score_answer(dirty, gold, answer_text)["total"])
