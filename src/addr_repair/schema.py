"""Fixed output contract: clean_record + changes + needs_review."""

from jsonschema import ValidationError, validate

OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["clean_record", "changes", "needs_review"],
    "additionalProperties": False,
    "properties": {
        "clean_record": {
            "type": "object",
            "required": ["name", "road", "house_number", "postcode", "locality", "country_code"],
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "road": {"type": "string"},
                "house_number": {"type": "string"},
                "postcode": {"type": "string"},
                "locality": {"type": "string"},
                "country_code": {"type": "string", "enum": ["DE"]},
            },
        },
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["field", "from", "to"],
                "additionalProperties": False,
                "properties": {
                    "field": {"type": "string"},
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                },
            },
        },
        "needs_review": {"type": "array", "items": {"type": "string"}},
    },
}


def validate_output(payload: dict) -> tuple[bool, str]:
    try:
        validate(instance=payload, schema=OUTPUT_SCHEMA)
    except ValidationError as exc:
        return False, str(exc.message)
    return True, "ok"
