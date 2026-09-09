"""Baseline gate: rules floor on the smoke fixture, scored when gold is available.

Usage:
    uv run python scripts/run_baseline.py --config configs/model.yaml
    uv run python scripts/run_baseline.py --pairs fixtures/smoke_100/pairs.jsonl \\
        --out evals/baseline-smoke/audit.jsonl

--fixture reads bare dirty records (no scoring). --pairs reads
{id, dirty, gold} lines, writes the same audit records plus a scored
metrics.json next to the audit output. The two inputs are mutually
exclusive.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, "src")

from addr_repair.audit import write_audit_record  # noqa: E402
from addr_repair.rules import repair_with_rules  # noqa: E402
from addr_repair.schema import validate_output, validate_output_semantics  # noqa: E402
from addr_repair.scorer import score_records  # noqa: E402

MODEL_REV = "rules-floor-v1"


def _read_input(path: Path, pairs: bool) -> list[dict]:
    """Return [{id, dirty, gold|None}] rows from bare or paired JSONL."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if pairs:
                rows.append(
                    {
                        "id": record.get("id", f"line-{line_number}"),
                        "dirty": record["dirty"],
                        "gold": record["gold"],
                    }
                )
            else:
                rows.append({"id": f"line-{line_number}", "dirty": record, "gold": None})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rules floor on smoke fixture.")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--fixture", default="fixtures/smoke_100/records.jsonl")
    parser.add_argument("--pairs", default=None, help="paired {id, dirty, gold} JSONL; enables scoring")
    parser.add_argument("--out", default="evals/baseline-smoke/audit.jsonl")
    args = parser.parse_args()

    if args.pairs and args.fixture != parser.get_default("fixture"):
        raise SystemExit("--pairs and --fixture are mutually exclusive")
    input_path = Path(args.pairs) if args.pairs else Path(args.fixture)
    if not input_path.exists():
        print(f"[run_baseline] input missing: {input_path} — run prepare_data first.")
        return
    scored = bool(args.pairs)
    n = 0
    score_rows = []
    out_path = Path(args.out)
    if out_path.exists():
        out_path.unlink()
    for row in _read_input(input_path, scored):
        dirty = row["dirty"]
        started = time.perf_counter()
        repaired, changes, needs_review = repair_with_rules(dirty)
        latency_ms = (time.perf_counter() - started) * 1000
        output = {"clean_record": repaired, "changes": changes, "needs_review": needs_review}
        ok, _ = validate_output(output)
        semantic_errors = validate_output_semantics(dirty, output)
        write_audit_record(
            out_path, dirty, output,
            model_rev=MODEL_REV, prompt_rev="n/a",
            schema_ok=ok, latency_ms=latency_ms,
            semantic_errors=semantic_errors,
        )
        if scored:
            score_rows.append(
                {
                    "dirty": dirty,
                    "gold": row["gold"],
                    "pred": repaired,
                    "schema_ok": ok,
                    "needs_review": needs_review,
                }
            )
        n += 1
    print(f"[run_baseline] wrote {n} audit records to {out_path}")
    if scored:
        metrics = score_records(score_rows)
        metrics_path = out_path.parent / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "system": "rules-floor",
                    "model_rev": MODEL_REV,
                    "input": str(input_path),
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "[run_baseline] rules-floor "
            f"precision={metrics['repair_precision']} recall={metrics['repair_recall']} "
            f"f1={metrics['repair_f1']} damage={metrics['damage_rate']} "
            f"schema={metrics['schema_validity']} -> {metrics_path}"
        )


if __name__ == "__main__":
    main()
