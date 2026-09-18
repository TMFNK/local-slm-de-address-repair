"""GRPO reward for address repair (plan 2026-09-16, Phase 1).

Format is a gate. Field outcomes carry the rank. ``scripts/train_grpo.py``
wraps :func:`reward_for_training` for TRL.

Gate: unparseable or schema-invalid -> 0. Valid JSON starts at +0.05.

Then:

- correct repair: +0.12 (non-empty dirty, pred == gold)
- honest review: +0.04 (empty critical field, kept empty, flagged)
- meaning_ok: +0.05 (internal consistency of ``changes``, not gold)
- missed repair: -0.04 (dirty != gold, pred == dirty, not an honest empty)
- wrong dirty edit: -0.12 (dirty != gold, pred != dirty, pred != gold)
- unsupported addition: -0.15 (scorer token-containment test)
- clean damage: -0.18 (dirty == gold, pred differs, not an addition)
- invention: -0.20 (empty dirty filled, even if pred == gold)
- spurious review flag: -0.02 (flagged field that is not an honest empty)

Passed-gate totals clip to [0.02, 1]. Failed-gate totals stay 0, so a
wrong-but-valid answer stays above broken JSON without the old +0.40
structure gift.

``n_wrong_dirty`` and ``n_missed`` stay in the breakdown. ``structure_total``
is gate + meaning only, for the reward-check shape-only picker.
"""

from __future__ import annotations

from addr_repair.parsing import extract_json
from addr_repair.schema import ABSTENTION_FIELDS, validate_output, validate_output_semantics
from addr_repair.scorer import FIELDS as SCORER_FIELDS
from addr_repair.scorer import _is_unsupported_addition, _norm

FIELDS = list(SCORER_FIELDS)
CRITICAL_FIELDS = tuple(ABSTENTION_FIELDS)

GATE_W = 0.05
MEANING_W = 0.05
CORRECT_W = 0.12
CLEAN_DAMAGE_W = -0.18
INVENTION_W = -0.20
ADDITION_W = -0.15
WRONG_DIRTY_W = -0.12
MISSED_W = -0.04
HONEST_REVIEW_W = 0.04
SPURIOUS_W = -0.02
PASSED_FLOOR = 0.02

WEIGHTS = {
    "gate": GATE_W,
    "meaning": MEANING_W,
    "correct_repair": CORRECT_W,
    "clean_damage": CLEAN_DAMAGE_W,
    "invention": INVENTION_W,
    "addition": ADDITION_W,
    "wrong_dirty": WRONG_DIRTY_W,
    "missed": MISSED_W,
    "honest_review": HONEST_REVIEW_W,
    "spurious_flag": SPURIOUS_W,
}


def _pred_record(payload: dict) -> dict:
    record = payload.get("clean_record")
    return record if isinstance(record, dict) else {}


def _zero_result(*, parsed_ok: bool, shape_ok: bool, semantic_errors: list[str]) -> dict:
    return {
        "total": 0.0,
        "structure_total": 0.0,
        "gate": 0.0,
        "parse": 0.0,
        "shape": 0.0,
        "meaning": 0.0,
        "repair_total": 0.0,
        "n_correct": 0,
        "n_clean_damage": 0,
        "n_invention": 0,
        "n_addition": 0,
        "n_wrong_dirty": 0,
        "n_missed": 0,
        "review_bonus": 0.0,
        "spurious_penalty": 0.0,
        "honest_review_fields": [],
        "spurious_flags": [],
        "parsed_ok": parsed_ok,
        "shape_ok": shape_ok,
        "meaning_ok": False,
        "semantic_errors": list(semantic_errors),
        "weights": dict(WEIGHTS),
    }


def score_parsed(dirty: dict, gold: dict | None, payload: dict) -> dict:
    """Score an already-parsed answer payload.

    ``gold`` may be None (repair parts are then 0; gate, meaning, and
    review parts still apply). Assumes the payload parsed. Use
    :func:`score_answer` for raw model text.
    """
    shape_ok, shape_error = validate_output(payload)
    if not shape_ok:
        return _zero_result(parsed_ok=True, shape_ok=False, semantic_errors=[shape_error])

    semantic_errors = validate_output_semantics(dirty, payload)
    meaning_ok = not semantic_errors

    pred = _pred_record(payload)
    reviews = payload.get("needs_review")
    review_list = list(reviews) if isinstance(reviews, list) else []

    n_correct = n_clean_damage = n_invention = n_addition = 0
    n_wrong_dirty = n_missed = 0
    if gold is not None:
        for field in FIELDS:
            d = _norm(dirty.get(field))
            g = _norm(gold.get(field))
            p = _norm(pred.get(field))
            d_empty = not d
            was_dirty = d != g
            repaired = p != d
            honest = not d and not p and field in CRITICAL_FIELDS and field in review_list
            if not repaired:
                if was_dirty and not honest:
                    n_missed += 1
                continue
            if d_empty:
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

    gate_part = GATE_W
    meaning_part = MEANING_W if meaning_ok else 0.0
    repair_total = (
        CORRECT_W * n_correct
        + CLEAN_DAMAGE_W * n_clean_damage
        + INVENTION_W * n_invention
        + ADDITION_W * n_addition
        + WRONG_DIRTY_W * n_wrong_dirty
        + MISSED_W * n_missed
    )
    review_bonus = HONEST_REVIEW_W * len(honest_fields)
    spurious_penalty = SPURIOUS_W * len(spurious_flags)
    raw = gate_part + meaning_part + repair_total + review_bonus + spurious_penalty
    total = max(PASSED_FLOOR, min(1.0, raw))

    return {
        "total": round(total, 4),
        "structure_total": round(gate_part + meaning_part, 4),
        "gate": round(gate_part, 4),
        "parse": round(gate_part, 4),
        "shape": round(gate_part, 4),
        "meaning": round(meaning_part, 4),
        "repair_total": round(repair_total, 4),
        "n_correct": n_correct,
        "n_clean_damage": n_clean_damage,
        "n_invention": n_invention,
        "n_addition": n_addition,
        "n_wrong_dirty": n_wrong_dirty,
        "n_missed": n_missed,
        "review_bonus": round(review_bonus, 4),
        "spurious_penalty": round(spurious_penalty, 4),
        "honest_review_fields": honest_fields,
        "spurious_flags": spurious_flags,
        "parsed_ok": True,
        "shape_ok": True,
        "meaning_ok": meaning_ok,
        "semantic_errors": list(semantic_errors),
        "weights": dict(WEIGHTS),
    }


def score_answer(dirty: dict, gold: dict | None, answer_text: str) -> dict:
    """Score one raw model answer string. Unparseable text scores 0 total."""
    payload, error = extract_json(answer_text or "")
    if error is not None or payload is None:
        return _zero_result(
            parsed_ok=False,
            shape_ok=False,
            semantic_errors=[error or "no-json-object-found"],
        )
    return score_parsed(dirty, gold, payload)


def reward_for_training(dirty: dict, gold: dict | None, answer_text: str) -> float:
    """Scalar reward in [0, 1] for the TRL GRPO loop."""
    return float(score_answer(dirty, gold, answer_text)["total"])
