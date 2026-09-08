"""Field-level scorer for repairs, abstentions, damage, and review.

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
    field_metrics = {}

    for field in FIELDS:
        field_metrics[field] = {
            "correct_repairs": 0,
            "missed_repairs": 0,
            "wrong_repairs": 0,
            "correct_abstentions": 0,
            "clean_field_damage": 0,
            "clean_fields": 0,
            "review_flagged": 0,
            "review_hits": 0,
        }

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
            metrics = field_metrics[field]
            if correct_repair:
                tp += 1
                metrics["correct_repairs"] += 1
            if repaired and not was_dirty:
                fp += 1  # touched a clean field
                metrics["clean_field_damage"] += 1
            if wrong_change and was_dirty:
                fp += 1
                metrics["wrong_repairs"] += 1
            if was_dirty and p != g:
                fn += 1
                if not repaired:
                    metrics["correct_abstentions"] += 1
                else:
                    metrics["missed_repairs"] += 1
            if not was_dirty:
                clean_total += 1
                metrics["clean_fields"] += 1
                if p != g:
                    damaged += 1
        flagged = row.get("needs_review", [])
        review_flagged += len(flagged)
        for field in flagged:
            if field not in field_metrics:
                continue
            review_total += 1
            field_metrics[field]["review_flagged"] += 1
            d = _norm(row["dirty"].get(field))
            if not d:
                review_hits += 1  # correctly refused to invent
                field_metrics[field]["review_hits"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = len(rows) or 1
    for metrics in field_metrics.values():
        correct = metrics["correct_repairs"]
        wrong = metrics["wrong_repairs"]
        metrics["repair_precision"] = round(correct / (correct + wrong) if correct + wrong else 0.0, 4)
        attempted = correct + metrics["missed_repairs"] + metrics["correct_abstentions"]
        metrics["repair_recall"] = round(correct / attempted if attempted else 0.0, 4)
        metrics["damage_rate"] = round(
            metrics["clean_field_damage"] / metrics["clean_fields"]
            if metrics["clean_fields"] else 0.0,
            4,
        )
        metrics["review_precision"] = round(
            metrics["review_hits"] / metrics["review_flagged"]
            if metrics["review_flagged"] else 0.0,
            4,
        )
        del metrics["review_hits"]

    return {
        "n": len(rows),
        "repair_precision": round(precision, 4),
        "repair_recall": round(recall, 4),
        "repair_f1": round(f1, 4),
        "damage_rate": round(damaged / clean_total, 4) if clean_total else 0.0,
        "review_precision": round(review_hits / review_total, 4) if review_total else 0.0,
        "review_coverage": round(review_flagged / (n * len(FIELDS)), 4),
        "schema_validity": round(schema_ok / n, 4),
        "by_field": field_metrics,
    }
