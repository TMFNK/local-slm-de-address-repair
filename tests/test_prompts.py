import sys

sys.path.insert(0, "src")

from addr_repair.prompts import FIELDS, PRODUCTION_PROMPT_REV, build_repair_prompt  # noqa: E402


def test_prompt_names_every_contract_field():
    messages = build_repair_prompt({field: "" for field in FIELDS})
    text = " ".join(message["content"] for message in messages)
    for field in (*FIELDS, "clean_record", "changes", "needs_review"):
        assert field in text


def test_prompt_forbids_fences_and_free_text():
    system = build_repair_prompt({})[0]["content"]
    assert "no markdown fences" in system.lower() or "no fences" in system.lower()


def test_prompt_rev_is_pinned_v2():
    assert PRODUCTION_PROMPT_REV == "v2"
