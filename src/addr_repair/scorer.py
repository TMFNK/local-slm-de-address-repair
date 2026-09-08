"""Field-level scorer: repair P/R/F1, damage rate, review, schema validity.

Compares predicted clean_record against paired gold clean_record,
given the dirty input. Exact string match after NFC + strip.
"""

import unicodedata

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def _norm(value: object) -> str:
    return unicodedata.normalize("NFC", str(value or "")).strip()


def score_records(rows: list[dict]) -> dict:
    """Each row: {dirty, gold, pred}. pred/gold are clean_record dicts."""
    tp = fp = fn = 0
    damaged = clean_total = 0
    review_hits = review_total = 0
    review_flagged = 0
    schema_ok = 0

    for row in rows:
        dirty, gold, pred = row["dirty"], row["gold"], row["pred"]
        if row.get("schema_ok"):
            schema_ok += 1
        for field in FIELDS:
            d, g, p = _norm(dirty.get(field)), _norm(gold.get(field)), _norm(pred.get(field))
            was_dirty = d != g
            repaired = p != d
            correct_repair = was_dirty and p == g
            wrong_change = repaired and p != g
            if correct_repair:
                tp += 1
            if repaired and not was_dirty:
                fp += 1  # touched a clean field
            if wrong_change and was_dirty:
                fp += 1
            if was_dirty and p != g:
                fn += 1
            if not was_dirty:
                clean_total += 1
                if p != g:
                    damaged += 1
        flagged = row.get("needs_review", [])
        review_flagged += len(flagged)
        for field in flagged:
            review_total += 1
            d, g = _norm(row["dirty"].get(field)), _norm(row["gold"].get(field))
            if not d and not g:
                review_hits += 1  # correctly refused to invent

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = len(rows) or 1
    return {
        "n": len(rows),
        "repair_precision": round(precision, 4),
        "repair_recall": round(recall, 4),
        "repair_f1": round(f1, 4),
        "damage_rate": round(damaged / clean_total, 4) if clean_total else 0.0,
        "review_precision": round(review_hits / review_total, 4) if review_total else 0.0,
        "review_coverage": round(review_flagged / (n * len(FIELDS)), 4),
        "schema_validity": round(schema_ok / n, 4),
    }
