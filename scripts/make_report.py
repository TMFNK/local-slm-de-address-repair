"""Build a Markdown summary from committed frozen-evaluation metrics.

Usage:
    uv run python scripts/make_report.py --evals evals/frozen-test-v3/
    uv run python scripts/make_report.py --evals evals/frozen-test-v3/ \
        --out paper/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEMS = ("rules", "base", "sft")


def build_report(evals_dir: Path) -> str:
    """Render the shared-boundary summary for one evaluation directory."""
    payloads = {}
    for system in SYSTEMS:
        path = evals_dir / system / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing committed metrics: {path}")
        payloads[system] = json.loads(path.read_text(encoding="utf-8"))

    boundary_keys = ("manifest", "manifest_records_sha256", "source_dirty_sha256", "source_clean_sha256")
    freezes = {
        tuple(payload["freeze"].get(key) for key in boundary_keys)
        for payload in payloads.values()
    }
    if len(freezes) != 1:
        manifests = {
            system: payload["freeze"].get("manifest_records_sha256")
            for system, payload in payloads.items()
        }
        raise ValueError(f"evaluation systems do not share one freeze boundary: {manifests}")

    freeze = payloads["rules"]["freeze"]
    lines = [
        "# Frozen evaluation report",
        "",
        (
            f"- Records: {freeze['manifest_records_sha256']} "
            f"({payloads['rules']['metrics']['n']:,} rows)"
        ),
        f"- Manifest: `{freeze['manifest']}`",
        "",
        "| System | Precision | Recall | F1 | Damage | Contract | Server errors |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {"rules": "Rules floor", "base": "Base MiniCPM5", "sft": "SFT"}
    for system in SYSTEMS:
        metrics = payloads[system]["metrics"]
        lines.append(
            f"| {names[system]} | {metrics['repair_precision']:.4f} | "
            f"{metrics['repair_recall']:.4f} | {metrics['repair_f1']:.4f} | "
            f"{metrics['damage_rate']:.4f} | {metrics['contract_validity']:.4f} | "
            f"{metrics.get('server_errors', 0)} |"
        )
    lines.extend(
        [
            "",
            (
                "This summary is generated from committed `metrics.json` files. "
                "Interpretation and historical status are documented in `docs/EVAL.md`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals", default="evals/frozen-test-v3/")
    parser.add_argument("--out", default=None, help="optional Markdown output path")
    args = parser.parse_args()
    report = build_report(Path(args.evals))
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
