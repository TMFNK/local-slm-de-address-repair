"""Fixed output contract plus consistency checks for model responses."""

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

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def validate_output(payload: dict) -> tuple[bool, str]:
    """Validate only the JSON Schema shape."""
    try:
        validate(instance=payload, schema=OUTPUT_SCHEMA)
    except ValidationError as exc:
        return False, str(exc.message)
    return True, "ok"


def validate_output_semantics(dirty: dict, payload: dict) -> list[str]:
    """Return consistency errors for a schema-valid response."""
    schema_ok, schema_error = validate_output(payload)
    if not schema_ok:
        return [f"schema: {schema_error}"]

    errors: list[str] = []
    clean_record = payload["clean_record"]
    changes = payload["changes"]
    reviews = payload["needs_review"]
    changed_fields = [change["field"] for change in changes]

    if len(changed_fields) != len(set(changed_fields)):
        errors.append("changes contains duplicate fields")
    if len(reviews) != len(set(reviews)):
        errors.append("needs_review contains duplicate fields")

    for change in changes:
        field = change["field"]
        if field not in FIELDS:
            errors.append(f"changes contains unknown field: {field}")
            continue
        dirty_value = str(dirty.get(field, "") or "")
        if change["from"] != dirty_value:
            errors.append(f"changes[{field}].from does not match dirty input")
        if change["to"] != clean_record[field]:
            errors.append(f"changes[{field}].to does not match clean_record")
        if change["from"] == change["to"]:
            errors.append(f"changes[{field}] records no actual change")

    for field in reviews:
        if field not in FIELDS:
            errors.append(f"needs_review contains unknown field: {field}")
            continue
        dirty_value = str(dirty.get(field, "") or "")
        if clean_record[field] != dirty_value:
            errors.append(f"needs_review[{field}] changes the value being reviewed")
        if field in changed_fields:
            errors.append(f"{field} appears in both changes and needs_review")

    for field in FIELDS:
        dirty_value = str(dirty.get(field, "") or "")
        if clean_record[field] != dirty_value and field not in changed_fields:
            errors.append(f"clean_record[{field}] changed without a change record")

    return errors
