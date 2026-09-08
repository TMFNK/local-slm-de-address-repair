"""Deterministic rules floor: whitespace, casing, Unicode, abbreviations.

Runs before any model call. Never invents missing values: empty or
unsupported fields are left alone for the review route.
"""

import re
import unicodedata

ABBREVIATIONS = {
    "str.": "straße",
    "str": "straße",
    "pl.": "platz",
}

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def normalize_whitespace(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def expand_abbreviations(road: str) -> str:
    # German street suffixes are glued to the stem (Musterstr. -> Musterstraße),
    # so match the "str." suffix at a token end, not a standalone word.
    # Note: no trailing \b after the period — "." is a non-word char.
    def _expand_suffix(text: str, short: str, full: str) -> str:
        stem = short.rstrip(".")
        pattern = re.compile(rf"(?i){re.escape(stem)}\.(?=\s|$)")

        def _replace(match: re.Match) -> str:
            word = full
            if match.group(0)[:1].isupper():
                word = full[:1].upper() + full[1:]
            return word

        return pattern.sub(_replace, text)

    result = road
    for short, full in ABBREVIATIONS.items():
        if short.startswith("str"):
            result = _expand_suffix(result, short, full)
        else:
            pattern = re.compile(rf"(?i)\b{re.escape(short)}(?=\s|$|[,.])")

            def _replace_word(match: re.Match, _full: str = full) -> str:
                word = _full
                if match.group(0)[:1].isupper():
                    word = _full[:1].upper() + _full[1:]
                return word

            result = pattern.sub(_replace_word, result)
    return result


def repair_with_rules(record: dict) -> tuple[dict, list[dict], list[str]]:
    """Return (repaired, changes, needs_review). Pure function, no I/O."""
    repaired: dict = {}
    changes: list[dict] = []
    needs_review: list[str] = []

    for field in FIELDS:
        raw = record.get(field, "")
        if raw is None:
            raw = ""
        raw = str(raw)
        if not raw.strip():
            repaired[field] = raw
            if field in ("road", "postcode", "house_number"):
                needs_review.append(field)
            continue
        cleaned = normalize_whitespace(raw)
        if field == "road":
            cleaned = expand_abbreviations(cleaned)
        if field == "country_code":
            cleaned = cleaned.upper()
        repaired[field] = cleaned
        if cleaned != raw:
            changes.append({"field": field, "from": raw, "to": cleaned})

    return repaired, changes, needs_review
