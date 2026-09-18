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


def _dirty_road():
    return {**BASE, "road": "Musterstr."}


def _gold():
    return dict(BASE)


def _correct_text():
    return _answer(
        dict(BASE),
        [{"field": "road", "from": "Musterstr.", "to": "Musterstraße"}],
        [],
    )


def _leave_alone_text():
    dirty = _dirty_road()
    return _answer(dict(dirty), [], [])


def _wrong_dirty_text():
    dirty = _dirty_road()
    pred = {**dirty, "road": "Hauptstraße"}
    return _answer(
        pred,
        [{"field": "road", "from": "Musterstr.", "to": "Hauptstraße"}],
        [],
    )


def test_correct_repair_scores_gate_plus_repair():
    result = score_answer(_dirty_road(), _gold(), _correct_text())
    assert result["n_correct"] == 1
    assert result["n_missed"] == 0
    assert result["repair_total"] == 0.12
    assert result["total"] == 0.22
    assert result["meaning_ok"] is True


def test_invented_correct_gold_fill_gets_no_credit_and_penalty():
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
    assert result["total"] == 0.02


def test_honest_keep_empty_plus_flag_gets_review_bonus():
    dirty = {**BASE, "road": ""}
    gold = {**BASE, "road": "Musterstraße"}
    text = _answer(dict(dirty), [], ["road"])
    result = score_answer(dirty, gold, text)
    assert result["n_invention"] == 0
    assert result["n_missed"] == 0
    assert result["review_bonus"] == 0.04
    assert result["honest_review_fields"] == ["road"]
    assert result["meaning_ok"] is True
    assert result["total"] == 0.14


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
    assert result["repair_total"] == -0.18
    assert result["total"] == 0.02


def test_unsupported_addition_gets_no_credit_and_penalty():
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
    assert result["total"] == 0.02


def test_flagging_nonempty_field_earns_no_bonus():
    dirty = dict(BASE)
    gold = dict(BASE)
    text = _answer(dict(BASE), [], ["road"])
    result = score_answer(dirty, gold, text)
    assert result["review_bonus"] == 0.0
    assert result["honest_review_fields"] == []
    assert result["spurious_flags"] == ["road"]
    assert result["spurious_penalty"] == -0.02


def test_broken_json_scores_zero_for_shape_and_meaning():
    result = score_answer(_dirty_road(), _gold(), "not json at all {{{")
    assert result["parsed_ok"] is False
    assert result["shape_ok"] is False
    assert result["meaning_ok"] is False
    assert result["total"] == 0.0


def test_wrong_dirty_edit_loses_to_leave_alone():
    dirty = _dirty_road()
    gold = _gold()
    leave = score_answer(dirty, gold, _leave_alone_text())
    wrong = score_answer(dirty, gold, _wrong_dirty_text())
    assert wrong["n_wrong_dirty"] == 1
    assert leave["n_missed"] == 1
    assert leave["total"] == 0.06
    assert wrong["total"] == 0.02
    assert wrong["total"] < leave["total"]


def test_leave_alone_loses_to_correct_repair():
    dirty = _dirty_road()
    gold = _gold()
    leave = score_answer(dirty, gold, _leave_alone_text())
    correct = score_answer(dirty, gold, _correct_text())
    assert leave["total"] < correct["total"]


def test_clean_damage_loses_to_noop_copy():
    dirty = dict(BASE)
    gold = dict(BASE)
    noop = score_answer(dirty, gold, _answer(dict(BASE), [], []))
    damaged = score_answer(
        dirty,
        gold,
        _answer(
            {**BASE, "road": "Falschweg"},
            [{"field": "road", "from": "Musterstraße", "to": "Falschweg"}],
            [],
        ),
    )
    assert noop["total"] == 0.10
    assert damaged["total"] < noop["total"]


def test_addition_loses_to_noop_copy():
    dirty = {**BASE, "name": "Helmholtz-Gymnasium"}
    gold = dict(dirty)
    noop = score_answer(dirty, gold, _answer(dict(dirty), [], []))
    added = score_answer(
        dirty,
        gold,
        _answer(
            {**dirty, "name": "Helmholtz-Gymnasium Karlsruhe"},
            [
                {
                    "field": "name",
                    "from": "Helmholtz-Gymnasium",
                    "to": "Helmholtz-Gymnasium Karlsruhe",
                }
            ],
            [],
        ),
    )
    assert added["n_addition"] == 1
    assert added["total"] < noop["total"]


def test_invented_gold_fill_loses_to_honest_flag():
    dirty = {**BASE, "postcode": ""}
    gold = {**BASE, "postcode": "44892"}
    honest = score_answer(dirty, gold, _answer(dict(dirty), [], ["postcode"]))
    invented = score_answer(
        dirty,
        gold,
        _answer(
            {**dirty, "postcode": "44892"},
            [{"field": "postcode", "from": "", "to": "44892"}],
            [],
        ),
    )
    assert invented["total"] < honest["total"]


def test_wrong_edit_with_matching_changes_still_loses():
    dirty = _dirty_road()
    gold = _gold()
    leave = score_answer(dirty, gold, _leave_alone_text())
    wrong = score_answer(dirty, gold, _wrong_dirty_text())
    assert wrong["meaning_ok"] is True
    assert wrong["total"] < leave["total"]


def test_broken_json_loses_to_valid_noop():
    dirty = dict(BASE)
    gold = dict(BASE)
    broken = score_answer(dirty, gold, "not json at all {{{")
    noop = score_answer(dirty, gold, _answer(dict(BASE), [], []))
    assert broken["total"] == 0.0
    assert noop["total"] > broken["total"]


def test_correct_repair_gap_beats_garbage_parse_gap():
    dirty = _dirty_road()
    gold = _gold()
    correct = score_answer(dirty, gold, _correct_text())["total"]
    wrong = score_answer(dirty, gold, _wrong_dirty_text())["total"]
    broken = score_answer(dirty, gold, "not json at all {{{")["total"]
    assert (correct - wrong) > (wrong - broken)
