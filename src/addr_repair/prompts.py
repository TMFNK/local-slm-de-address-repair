"""Pinned production prompt. Rev changes go through configs/model.yaml."""

PRODUCTION_PROMPT_REV = "v2"

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]

SYSTEM = (
    "Repair the German address record. Fix whitespace, casing, Unicode, "
    "punctuation, documented abbreviations, and obvious spelling or field "
    "placement errors. Never invent a missing road, postcode, or house number: "
    "list such fields in needs_review. "
    "Reply with exactly one JSON object and nothing else: no markdown fences, "
    "no explanation. The object has exactly three keys: clean_record, "
    "changes, needs_review. clean_record is an object with exactly these "
    "string keys: name, road, house_number, postcode, locality, "
    "country_code. changes is a list of {field, from, to} objects, one per "
    "field you changed. needs_review is a list of field names."
)


def build_repair_prompt(dirty: dict) -> list[dict]:
    shown = "\n".join(f"{k}: {dirty.get(k, '')}" for k in FIELDS)
    skeleton = (
        '{"clean_record": {"name": "...", "road": "...", "house_number": "...", '
        '"postcode": "...", "locality": "...", "country_code": "..."}, '
        '"changes": [{"field": "...", "from": "...", "to": "..."}], '
        '"needs_review": ["..."]}'
    )
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "Dirty record:\n" + shown + "\n\nReturn JSON with keys "
                "clean_record, changes, needs_review. "
                f"Follow this exact shape:\n{skeleton}"
            ),
        },
    ]
