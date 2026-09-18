"""Field-level scorer for repairs, abstentions, damage, and review.

Compares predicted clean_record against paired gold clean_record,
given the dirty input. Exact string match after NFC + strip.

Fix 3 (2026-09-09) and Fix 4 (2026-09-11) — evidence-safe repair + validity split:
- A correct repair (TP) requires evidence: the dirty field must be
  non-empty. Filling an empty field is an empty-field fill, even when the
  guess matches gold (geocoder gold is invisible at inference time).
- An empty-field fill (dirty empty + pred filled) counts as one FP per field
  and is reported under the historical keys ``inventions`` /
  ``invention_rate`` plus per-field ``inventions``. It never counts as TP,
  with or without a review flag.
- An unsupported addition retains a non-empty dirty value as a contiguous
  token sequence while adding extra tokens. It is reported separately as
  ``unsupported_additions`` / ``unsupported_addition_rate``.
- Correct abstention (dirty empty, gold filled, pred kept empty) still
  counts as a recall miss (FN). This is the Step 5 contract: the test set
  and scorer stay unchanged on abstention-vs-recall, and the final report
  must own the tension in one explicit paragraph.
- Validity split: ``schema_validity`` (JSON Schema only), plus new
  ``semantic_validity`` (no semantic errors), ``parse_rate`` (output
  parsed), and ``contract_validity`` (schema AND semantics — the usable
  output rate). Repair scoring is NOT gated on validity here; measurement
  first, gating is a later policy decision.
- Row keys: {dirty, gold, pred, needs_review?, schema_ok?, semantic_ok?
  or semantic_errors?, parsed?}. Missing ``schema_ok``/``semantic_ok``
  counts as invalid (same convention as before); missing ``parsed``
  defaults to True (pre-Fix-3 rows were all parsed outputs).
"""

import unicodedata

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def _norm(value: object) -> str:
    return unicodedata.normalize("NFC", str(value or "")).strip()


def _semantic_ok(row: dict) -> bool:
    if "semantic_ok" in row:
        return bool(row["semantic_ok"])
    if "semantic_errors" in row:
        return not row["semantic_errors"]
    return False


def _is_unsupported_addition(dirty: str, pred: str) -> bool:
    """Detect a prediction that extends a non-empty dirty value."""
    if not dirty or pred == dirty:
        return False
    dirty_tokens = dirty.casefold().split()
    pred_tokens = pred.casefold().split()
    if not dirty_tokens or len(pred_tokens) <= len(dirty_tokens):
        return False
    width = len(dirty_tokens)
    return any(
        pred_tokens[index : index + width] == dirty_tokens
        for index in range(len(pred_tokens) - width + 1)
    )


def score_records(rows: list[dict]) -> dict:
    """Each row: {dirty, gold, pred}. pred/gold are clean_record dicts."""
    tp = fp = fn = 0
    damaged = clean_total = 0
    review_hits = review_total = 0
    review_flagged = 0
    review_records = 0
    schema_ok = 0
    semantic_ok = 0
    parsed_ok = 0
    contract_ok = 0
    inventions = 0
    unsupported_additions = 0
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
            "inventions": 0,
            "unsupported_additions": 0,
        }

    for row in rows:
        dirty, gold, pred = row["dirty"], row["gold"], row["pred"]
        row_schema_ok = bool(row.get("schema_ok"))
        row_semantic_ok = _semantic_ok(row)
        row_parsed = bool(row.get("parsed", True))
        if row_schema_ok:
            schema_ok += 1
        if row_semantic_ok:
            semantic_ok += 1
        if row_parsed:
            parsed_ok += 1
        if row_schema_ok and row_semantic_ok:
            contract_ok += 1
        for field in FIELDS:
            d, g, p = _norm(dirty.get(field)), _norm(gold.get(field)), _norm(pred.get(field))
            was_dirty = d != g
            repaired = p != d
            d_empty = not d
            invented = d_empty and repaired
            unsupported_addition = _is_unsupported_addition(d, p)
            correct_repair = was_dirty and p == g and not d_empty
            metrics = field_metrics[field]
            if invented:
                # Filled a field with no evidence: always FP, never TP,
                # regardless of gold match or review flag.
                fp += 1
                inventions += 1
                metrics["inventions"] += 1
            if unsupported_addition:
                unsupported_additions += 1
                metrics["unsupported_additions"] += 1
            elif repaired and not was_dirty:
                fp += 1  # touched a clean field
                metrics["clean_field_damage"] += 1
            elif repaired and p != g:
                fp += 1
                metrics["wrong_repairs"] += 1
            if correct_repair:
                tp += 1
                metrics["correct_repairs"] += 1
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
        if flagged:
            review_records += 1
        for field in flagged:
            if not isinstance(field, str):
                continue
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
        "review_field_rate": round(review_flagged / (n * len(FIELDS)), 4),
        "review_record_coverage": round(review_records / n, 4) if rows else 0.0,
        "schema_validity": round(schema_ok / n, 4),
        "semantic_validity": round(semantic_ok / n, 4),
        "parse_rate": round(parsed_ok / n, 4),
        "contract_validity": round(contract_ok / n, 4),
        "inventions": inventions,
        "invention_rate": round(inventions / (n * len(FIELDS)), 4),
        "unsupported_additions": unsupported_additions,
        "unsupported_addition_rate": round(
            unsupported_additions / (n * len(FIELDS)), 4
        ),
        "by_field": field_metrics,
    }
