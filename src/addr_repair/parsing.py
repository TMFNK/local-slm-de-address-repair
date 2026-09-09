"""Response parsing for model output. Raw text in, contract-checked dict out.

Base models do not obey the output contract out of the box: they wrap
replies in code fences or return `clean_record` as a flat string. This
module isolates that mess so every system scores through one path.
"""

import json

from addr_repair.scorer import FIELDS

REQUIRED_KEYS = ("clean_record", "changes", "needs_review")


def extract_json(text: str) -> tuple[dict | None, str | None]:
    """Return (parsed object, error). Error is None on success."""
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        lines = lines[1:]  # drop opening fence (``` or ```json)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            return None, "no-json-object-found"
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, f"malformed-json: {exc}"
    if not isinstance(parsed, dict):
        return None, "top-level-json-is-not-an-object"
    return parsed, None


def check_contract(parsed: dict) -> list[str]:
    """Return contract violations. Empty list means the shape is usable."""
    errors = []
    for key in REQUIRED_KEYS:
        if key not in parsed:
            errors.append(f"missing-key: {key}")
    record = parsed.get("clean_record")
    if "clean_record" in parsed and not isinstance(record, dict):
        errors.append("clean_record-is-not-an-object")
    elif isinstance(record, dict):
        for field in FIELDS:
            if field not in record:
                errors.append(f"clean_record-missing-field: {field}")
    if "changes" in parsed and not isinstance(parsed["changes"], list):
        errors.append("changes-is-not-a-list")
    if "needs_review" in parsed and not isinstance(parsed["needs_review"], list):
        errors.append("needs_review-is-not-a-list")
    return errors


def parse_response(text: str) -> tuple[dict | None, list[str]]:
    """Parse raw model text into a contract-checked output, plus errors."""
    parsed, error = extract_json(text)
    if error is not None:
        return None, [error]
    errors = check_contract(parsed)
    if errors:
        return None, errors
    return parsed, []
