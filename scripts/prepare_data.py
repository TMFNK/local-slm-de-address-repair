"""Prepare data: verify archive, write manifests, build smoke fixture.

Usage:
    uv run python scripts/prepare_data.py --config configs/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify source archive and write split manifests.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()
    raw = Path(args.raw_dir)
    print(f"[prepare_data] config={args.config} raw_dir={raw}")
    print("[prepare_data] TODO: checksum archive, split entity ids, write data/manifests/, build fixtures/smoke_100/")
    print("[prepare_data] Test set must use original paired dirty/clean rows only.")


if __name__ == "__main__":
    main()
