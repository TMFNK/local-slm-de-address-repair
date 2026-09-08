"""Build metrics.json + failure exhibits + report stub from frozen evals.

Usage:
    uv run python scripts/make_report.py --evals evals/frozen-test/
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report artifacts.")
    parser.add_argument("--evals", default="evals/frozen-test/")
    args = parser.parse_args()
    print(f"[make_report] evals={args.evals}")
    print("[make_report] TODO: aggregate rules/base/sft metrics, pick 2 failure exhibits, write paper/report.md.")


if __name__ == "__main__":
    main()
