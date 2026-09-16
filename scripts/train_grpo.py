"""GRPO on top of v3 SFT (plan 2026-09-15, step 5). Colab-safe.

Usage (Colab GPU, after staging the v3 adapter — step 6 notebook):
    !python scripts/train_grpo.py --config configs/train_grpo.yaml

Dry run (no GPU, no downloads, no Drive — safe anywhere):
    uv run python scripts/train_grpo.py --config configs/train_grpo.yaml --dry-run

What the run does: starts from the frozen v3 LoRA adapter, rolls out groups
of 4 answers per dirty record at temperature 0.7, and rewards them with
``rewards.reward_for_training`` (structure + per-field repair - damage -
fills - additions + honest review, clipped to [0, 1]). Training rows are
dirty-needs-repair train rows only (seeded shuffle, first N) — no-op rows
teach nothing and are excluded by construction.

Checkpoint choice (plan step 6 rule): post-hoc scoring of retained
checkpoints on a val sample, winner = max mean(review_precision, 1-damage,
F1) with damage and fill counts as guards, never recall alone. Reuses the
SFT selection helpers so both stages pick by the same rule.

Prompt and template parity with SFT: prompts come from
``prompts.build_repair_prompt`` and the tokenizer gets the same
training-only MiniCPM recipe template (never saved, never served).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import train_sft

from addr_repair.prompts import build_repair_prompt
from addr_repair.rewards import WEIGHTS, reward_for_training
from addr_repair.scorer import FIELDS

V3_ADAPTER_HASH = "7103451b9cc916920c59e5d68e8c3ada852b294be5eab55a0d1a9abdc77ec712"


def select_grpo_rows(
    pairs: list[dict], n: int, seed: int
) -> tuple[list[dict], dict]:
    """Pick hard train rows: dirty differs from gold, seeded shuffle, first n.

    No-op rows (dirty already equals gold everywhere) carry no repair signal
    for a reward based on fixes, so they are excluded before the shuffle.
    Deterministic in the seed; records the shuffle for the training record.
    """
    hard = [
        pair
        for pair in pairs
        if any(str(pair["dirty"].get(f) or "") != str(pair["gold"].get(f) or "") for f in FIELDS)
    ]
    order = list(range(len(hard)))
    random.Random(seed).shuffle(order)
    picked = [hard[i] for i in order[: max(0, n)]]
    rows = [
        {
            "id": pair["id"],
            "prompt": build_repair_prompt(pair["dirty"]),
            "dirty": pair["dirty"],
            "gold": pair["gold"],
        }
        for pair in picked
    ]
    stats = {
        "train_pairs": len(pairs),
        "hard_pairs": len(hard),
        "picked": len(rows),
        "requested": n,
        "seed": seed,
    }
    return rows, stats


def _completion_to_text(completion) -> str:
    """Unwrap one TRL completion to raw text.

    TRL 1.12 hands conversational rollouts to reward funcs as message
    lists (``[[{"role": "assistant", "content": ...}]]``), not strings.
    Plain strings pass through untouched.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    if isinstance(completion, list):
        parts = []
        for item in completion:
            items = item if isinstance(item, list) else [item]
            for sub in items:
                if isinstance(sub, dict):
                    parts.append(str(sub.get("content") or ""))
                else:
                    parts.append(str(sub or ""))
        return "".join(parts)
    return str(completion or "")


def grpo_reward(prompts, completions, dirty, gold, **kwargs) -> list[float]:
    """TRL reward func: one scalar per completion (extra columns ignored)."""
    del prompts, kwargs
    texts = [_completion_to_text(c) for c in completions]
    return [reward_for_training(d, g, t) for d, g, t in zip(dirty, gold, texts)]


def build_grpo_config(train_cfg: dict, output_dir: Path):
    """Build the TRL GRPOConfig, dropping (and reporting) unsupported kwargs.

    Same guard as the SFT path: the dry-run never constructs this object, so
    a drifted field name would only explode on the paid GPU. Covered by a
    test that really constructs GRPOConfig. Returns (args, dropped_kwargs).
    """
    from trl import GRPOConfig

    wanted = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "warmup_steps": train_cfg.get("warmup_steps", 10),
        "max_steps": train_cfg.get("max_steps", 150),
        "bf16": bool(train_cfg.get("bf16", False)),
        "fp16": bool(train_cfg.get("fp16", True)),
        "temperature": train_cfg.get("temperature", 0.7),
        "num_generations": train_cfg.get("num_generations", 4),
        "max_completion_length": train_cfg.get("max_completion_length", 512),
        "beta": train_cfg.get("beta", 0.01),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", True)),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "save_steps": train_cfg.get("save_steps", 50),
        "save_total_limit": train_cfg.get("save_total_limit", 3),
        "report_to": train_cfg.get("report_to", "none"),
        "seed": train_cfg.get("seed", 7),
        "remove_unused_columns": False,
    }
    supported = set(GRPOConfig.__dataclass_fields__)
    dropped = sorted(k for k in wanted if k not in supported)
    kept = {k: v for k, v in wanted.items() if k in supported}
    return GRPOConfig(**kept), dropped


def run_dry_run(resolved: dict) -> int:
    """Log the full GRPO plan without GPU, downloads, or Drive. Returns exit code."""
    train_sft.assert_prompt_parity(resolved)
    train_pairs, train_manifest = train_sft.load_split_pairs(resolved, "train")
    val_pairs, val_manifest = train_sft.load_split_pairs(resolved, "val")
    train_cfg = resolved["train"]
    _rows, stats = select_grpo_rows(
        train_pairs, int(train_cfg.get("train_rows", 1500)), int(train_cfg.get("train_seed", 7))
    )
    grpo_args, dropped = build_grpo_config(train_cfg, Path(train_cfg["output_dir"]))
    prompts_per_step = (
        train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"]
    )
    print("[train_grpo] DRY RUN — no GPU, no downloads, no writes")
    print(f"[train_grpo] model={resolved['model']['model_id']} rev={resolved['model']['model_rev']}")
    print(f"[train_grpo] init_adapter={train_cfg.get('init_adapter_path')} (staged by notebook; not checked here)")
    print(f"[train_grpo] prompt_rev={resolved['model']['prompt_rev']} (code {train_sft.PRODUCTION_PROMPT_REV})")
    print(
        f"[train_grpo] train n={len(train_pairs)} hash={train_manifest['records_sha256'][:16]} "
        f"val n={len(val_pairs)} hash={val_manifest['records_sha256'][:16]}"
    )
    print(
        f"[train_grpo] grpo rows: hard={stats['hard_pairs']}/{stats['train_pairs']} "
        f"picked={stats['picked']} seed={stats['seed']}"
    )
    print(f"[train_grpo] reward weights={json.dumps(WEIGHTS, sort_keys=True)}")
    print(
        f"[train_grpo] group={grpo_args.num_generations} temp={grpo_args.temperature} "
        f"max_completion={grpo_args.max_completion_length} beta={grpo_args.beta} "
        f"lr={grpo_args.learning_rate}"
    )
    print(
        f"[train_grpo] max_steps={grpo_args.max_steps} (~{prompts_per_step} prompts/step, "
        f"~{prompts_per_step * grpo_args.num_generations} completions/step)"
    )
    print(
        f"[train_grpo] lora r={resolved['lora']['r']} alpha={resolved['lora']['lora_alpha']} "
        "(continued from v3 adapter, LoRA only)"
    )
    if dropped:
        print(f"[train_grpo] WARNING: GRPOConfig ignores unsupported args: {dropped}")
    print(
        f"[train_grpo] val selection: {train_cfg.get('val_samples', 100)} samples, "
        "winner = max mean(review_precision, 1-damage, F1), ties by contract then fewest inventions"
    )
    print(f"[train_grpo] output_dir={train_cfg['output_dir']}")
    return 0


def assert_adapter_parity(active, lora_cfg: dict, adapter_path: Path) -> None:
    """Fail fast unless the staged v3 adapter matches the pinned LoRA shape.

    ``active`` is the loaded adapter's PeftConfig (``model.peft_config["default"]``).
    TRL forbids PeftModel + peft_config together, so the run continues this
    adapter in place — it must be the v3 shape, never a foreign one.
    """
    expected_targets = sorted(lora_cfg["target_modules"])
    actual_targets = sorted(active.target_modules)
    if (
        active.r != lora_cfg["r"]
        or active.lora_alpha != lora_cfg["lora_alpha"]
        or actual_targets != expected_targets
    ):
        raise SystemExit(
            f"staged adapter {adapter_path} is r={active.r} alpha={active.lora_alpha} "
            f"targets={actual_targets}; configs/lora.yaml pins r={lora_cfg['r']} "
            f"alpha={lora_cfg['lora_alpha']} targets={expected_targets}. "
            "Refusing to continue a foreign adapter."
        )
    print(
        f"[train_grpo] continuing v3 adapter in place "
        f"(r={active.r} alpha={active.lora_alpha}); no fresh peft_config per TRL rule"
    )


def run_training(resolved: dict, resume: bool = False) -> dict:
    """Brief GRPO run from the v3 adapter, then post-hoc val selection."""
    import torch
    import transformers
    from datasets import Dataset
    from peft import PeftModel
    from trl import GRPOTrainer

    train_sft.assert_prompt_parity(resolved)
    train_cfg, model_cfg, lora_cfg = resolved["train"], resolved["model"], resolved["lora"]
    started = time.perf_counter()
    transformers.set_seed(train_cfg.get("seed", 7))

    train_pairs, train_manifest = train_sft.load_split_pairs(resolved, "train")
    val_pairs, val_manifest = train_sft.load_split_pairs(resolved, "val")
    rows, stats = select_grpo_rows(
        train_pairs, int(train_cfg.get("train_rows", 1500)), int(train_cfg.get("train_seed", 7))
    )

    adapter_path = Path(str(train_cfg.get("init_adapter_path", "")))
    if not adapter_path.is_dir():
        raise SystemExit(
            f"v3 adapter not found at {adapter_path}; the step-6 notebook must stage it "
            f"(adapter hash {V3_ADAPTER_HASH}) before training."
        )
    model_id = model_cfg["model_id"]
    model_rev = model_cfg["model_rev"]
    trust_remote = bool(train_cfg.get("trust_remote_code", False))
    use_bf16 = bool(train_cfg.get("bf16", False))
    use_fp16 = bool(train_cfg.get("fp16", True))
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=model_rev, trust_remote_code=trust_remote, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    template_before = train_sft.apply_training_template(tokenizer)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_rev,
        trust_remote_code=trust_remote,
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)
    assert_adapter_parity(model.peft_config["default"], lora_cfg, adapter_path)
    output_dir = Path(train_cfg["output_dir"])
    grpo_args, dropped = build_grpo_config(train_cfg, output_dir)
    if dropped:
        print(f"[train_grpo] WARNING: GRPOConfig ignores unsupported args: {dropped}")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward,
        args=grpo_args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(        yaml.safe_dump(
            {"train": train_cfg, "model": model_cfg, "lora": lora_cfg, "data": resolved["data"]},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    resume_from = train_sft.resolve_resume_checkpoint(output_dir, resume)
    if resume and resume_from is None:
        print("[train_grpo] --resume set but no checkpoint found; starting fresh")
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    train_seconds = round(time.perf_counter() - started, 1)

    del trainer
    del model
    gc.collect()
    train_sft._empty_cuda_cache()

    val_samples = int(train_cfg.get("val_samples", 100))
    val_gen_tokens = int(train_cfg.get("val_gen_max_tokens", 256))
    scored: list[tuple[str, dict]] = []
    val_started = time.perf_counter()
    for checkpoint in train_sft._iter_candidates(output_dir):
        metrics = train_sft.score_candidate(
            checkpoint, output_dir, model_id, model_rev, trust_remote, use_bf16,
            tokenizer, val_pairs, val_samples, val_gen_tokens,
        )
        if metrics is None:
            continue
        train_sft.write_score_file(output_dir, checkpoint.name, metrics)
        scored.append((checkpoint.name, metrics))
    val_seconds = round(time.perf_counter() - val_started, 1)
    winner_name, winner_metrics = train_sft.pick_best(scored)

    record = {
        "mode": "grpo-v4",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "model_id": model_id,
        "model_rev": model_rev,
        "init_adapter_path": str(adapter_path),
        "v3_adapter_hash": V3_ADAPTER_HASH,
        "continued_adapter_in_place": True,
        "prompt_rev": model_cfg.get("prompt_rev"),
        "training_template_source": train_sft.TRAIN_TEMPLATE_SOURCE,
        "inference_template_sha256": hashlib.sha256(
            (template_before or "").encode("utf-8")
        ).hexdigest(),
        "reward_weights": dict(WEIGHTS),
        "num_generations": grpo_args.num_generations,
        "temperature": grpo_args.temperature,
        "max_completion_length": grpo_args.max_completion_length,
        "beta": grpo_args.beta,
        "learning_rate": grpo_args.learning_rate,
        "max_steps": grpo_args.max_steps,
        "dropped_grpo_kwargs": dropped,
        "train_manifest_sha256": train_manifest["records_sha256"],
        "val_manifest_sha256": val_manifest["records_sha256"],
        "grpo_row_stats": stats,
        "train_seconds": train_seconds,
        "val_scoring_seconds": val_seconds,
        "checkpoints": [
            {"name": name, "selection_score": train_sft.selection_score(m), "metrics": m}
            for name, m in scored
        ],
        "winner": winner_name,
        "winner_selection_score": train_sft.selection_score(winner_metrics),
        "selection_rule": "max mean(review_precision, 1-damage_rate, repair_f1); ties by contract_validity then fewest inventions",
        "stop_rule": "stop if damage or empty fills rise above v3, even if F1 rises",
        "gpu": train_sft._gpu_record(),
        "platform": f"{platform.system()} {platform.machine()}",
        "repo_commit": train_sft._repo_commit(),
    }
    (output_dir / "training_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "selection.json").write_text(
        json.dumps(
            {"winner": winner_name, "selection_score": train_sft.selection_score(winner_metrics),
             "metrics": winner_metrics, "rule": record["selection_rule"]},
            ensure_ascii=False, indent=2, sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[train_grpo] done in {train_seconds}s; winner={winner_name} "
          f"score={train_sft.selection_score(winner_metrics)} -> {output_dir}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Brief GRPO on v3 SFT adapter (LoRA only).")
    parser.add_argument("--config", default="configs/train_grpo.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Log plan without training.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue from the latest checkpoint in output_dir (Colab recovery).",
    )
    args = parser.parse_args()
    resolved = train_sft.load_resolved_config(args.config)
    if args.dry_run:
        raise SystemExit(run_dry_run(resolved))
    run_training(resolved, resume=args.resume)


if __name__ == "__main__":
    main()
