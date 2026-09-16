import json
import sys

sys.path.insert(0, "src")

from addr_repair.rewards import score_answer

BASE = {
    "name": "x",
    "road": "Musterstraße",
    "house_number": "12",
    "postcode": "80331",
    "locality": "München",
    "country_code": "DE",
}


def _answer(pred: dict, changes: list, review: list) -> str:
    return json.dumps(
        {"clean_record": pred, "changes": changes, "needs_review": review},
        ensure_ascii=False,
    )


def test_correct_repair_scores_structure_plus_repair():
    dirty = {**BASE, "road": "Musterstr."}
    gold = dict(BASE)
    text = _answer(
        dict(BASE),
        [{"field": "road", "from": "Musterstr.", "to": "Musterstraße"}],
        [],
    )
    result = score_answer(dirty, gold, text)
    assert result["n_correct"] == 1
    assert result["repair_total"] == 0.10
    assert result["total"] == 0.50
    assert result["meaning_ok"] is True


def test_invented_correct_gold_fill_gets_no_credit_and_penalty():
    # Fix 3 policy: dirty '' -> gold value is a fill, never a repair,
    # even when the guess matches gold.
    dirty = {**BASE, "postcode": ""}
    gold = {**BASE, "postcode": "44892"}
    pred = {**dirty, "postcode": "44892"}
    text = _answer(
        pred,
        [{"field": "postcode", "from": "", "to": "44892"}],
        [],
    )
    result = score_answer(dirty, gold, text)
    assert result["n_correct"] == 0
    assert result["n_invention"] == 1
    assert result["repair_total"] == -0.20
    assert result["meaning_ok"] is False
    assert result["total"] == 0.0


def test_honest_keep_empty_plus_flag_gets_review_bonus():
    dirty = {**BASE, "road": ""}
    gold = {**BASE, "road": "Musterstraße"}
    text = _answer(dict(dirty), [], ["road"])
    result = score_answer(dirty, gold, text)
    assert result["n_invention"] == 0
    assert result["review_bonus"] == 0.05
    assert result["honest_review_fields"] == ["road"]
    assert result["meaning_ok"] is True
    assert result["total"] == 0.45


def test_clean_field_damage_gets_penalty():
    dirty = dict(BASE)
    gold = dict(BASE)
    pred = {**BASE, "road": "Falschweg"}
    text = _answer(
        pred,
        [{"field": "road", "from": "Musterstraße", "to": "Falschweg"}],
        [],
    )
    result = score_answer(dirty, gold, text)
    assert result["n_correct"] == 0
    assert result["n_clean_damage"] == 1
    assert result["repair_total"] == -0.15
    assert result["total"] == 0.25


def test_unsupported_addition_gets_no_credit_and_penalty():
    # Plan exhibit: well-formed but wrong, passes schema and meaning.
    dirty = {**BASE, "name": "Helmholtz-Gymnasium"}
    gold = dict(dirty)
    pred = {**dirty, "name": "Helmholtz-Gymnasium Karlsruhe"}
    text = _answer(
        pred,
        [
            {
                "field": "name",
                "from": "Helmholtz-Gymnasium",
                "to": "Helmholtz-Gymnasium Karlsruhe",
            }
        ],
        [],
    )
    result = score_answer(dirty, gold, text)
    assert result["shape_ok"] is True
    assert result["meaning_ok"] is True
    assert result["n_correct"] == 0
    assert result["n_addition"] == 1
    assert result["repair_total"] == -0.15
    assert result["total"] == 0.25


def test_flagging_nonempty_field_earns_no_bonus():
    dirty = dict(BASE)
    gold = dict(BASE)
    text = _answer(dict(BASE), [], ["road"])
    result = score_answer(dirty, gold, text)
    assert result["review_bonus"] == 0.0
    assert result["honest_review_fields"] == []
    assert result["spurious_flags"] == ["road"]


def test_broken_json_scores_zero_for_shape_and_meaning():
    dirty = {**BASE, "road": "Musterstr."}
    gold = dict(BASE)
    result = score_answer(dirty, gold, "not json at all {{{")
    assert result["parsed_ok"] is False
    assert result["shape_ok"] is False
    assert result["meaning_ok"] is False
    assert result["total"] == 0.0
