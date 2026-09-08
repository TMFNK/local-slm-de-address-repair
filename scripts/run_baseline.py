"""Baseline gate: rules floor + base MiniCPM5 on the smoke fixture.

Usage:
    uv run python scripts/run_baseline.py --config configs/model.yaml
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rules floor on smoke fixture.")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--fixture", default="fixtures/smoke_100/records.jsonl")
    parser.add_argument("--out", default="evals/baseline-smoke/audit.jsonl")
    args = parser.parse_args()

    fixture = Path(args.fixture)
    if not fixture.exists():
        print(f"[run_baseline] fixture missing: {fixture} — run prepare_data first.")
        return
    n = 0
    with fixture.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            dirty = json.loads(line)
            started = time.perf_counter()
            repaired, changes, needs_review = repair_with_rules(dirty)
            latency_ms = (time.perf_counter() - started) * 1000
            output = {"clean_record": repaired, "changes": changes, "needs_review": needs_review}
            ok, _ = validate_output(output)
            semantic_errors = validate_output_semantics(dirty, output)
            write_audit_record(
                args.out, dirty, output,
                model_rev="rules-floor-v1", prompt_rev="n/a",
                schema_ok=ok, latency_ms=latency_ms,
                semantic_errors=semantic_errors,
            )
            n += 1
    print(f"[run_baseline] wrote {n} audit records to {args.out}")


if __name__ == "__main__":
    main()
