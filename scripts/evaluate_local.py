"""Frozen local eval via llama.cpp. Same runtime for base + tuned.

Usage:
    uv run python scripts/evaluate_local.py --config configs/model.yaml --system rules|sft|base
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen held-out evaluation.")
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--system", default="rules", choices=["rules", "base", "sft"])
    parser.add_argument("--test", default="data/manifests/test.json")
    parser.add_argument("--out-dir", default="evals/frozen-test/")
    args = parser.parse_args()
    print(f"[evaluate_local] system={args.system} config={args.config}")
    print("[evaluate_local] TODO: call llama.cpp server, validate, audit, score, save metrics.json.")
    print("[evaluate_local] FREEZE: do not change prompt, normalizer, model, or test data after this.")


if __name__ == "__main__":
    main()
