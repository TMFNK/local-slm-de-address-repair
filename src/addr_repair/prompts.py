"""Pinned production prompt. Rev changes go through configs/model.yaml."""

PRODUCTION_PROMPT_REV = "v1"

SYSTEM = (
    "Repair the German address record. Fix whitespace, casing, Unicode, "
    "punctuation, documented abbreviations, and obvious spelling or field "
    "placement errors. Never invent a missing road, postcode, or house number: "
    "list such fields in needs_review. Reply with JSON only."
)


def build_repair_prompt(dirty: dict) -> list[dict]:
    fields = ["name", "road", "house_number", "postcode", "locality", "country_code"]
    shown = "\n".join(f"{k}: {dirty.get(k, '')}" for k in fields)
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                "Dirty record:\n" + shown + "\n\nReturn JSON with keys "
                "clean_record, changes, needs_review."
            ),
        },
    ]
