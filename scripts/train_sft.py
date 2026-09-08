"""SFT entrypoint, Colab-safe. Notebook calls main() with the same config.

Usage (Colab):
    !python scripts/train_sft.py --config configs/train_colab.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for MiniCPM5-1B address repair.")
    parser.add_argument("--config", default="configs/train_colab.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Log plan without training.")
    args = parser.parse_args()
    print(f"[train_sft] config={args.config} dry_run={args.dry_run}")
    print("[train_sft] TODO: load chat messages, TRL + PEFT LoRA r16, assistant-only loss,")
    print("[train_sft] select checkpoint on validation only, save hashes + GPU + time record.")


if __name__ == "__main__":
    main()
