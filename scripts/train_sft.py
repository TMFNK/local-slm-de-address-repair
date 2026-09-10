"""SFT entrypoint, Colab-safe. Notebook calls main() with the same config.

Usage (Colab GPU):
    !python scripts/train_sft.py --config configs/train_colab.yaml

Dry run (no GPU, no model download — safe anywhere):
    uv run python scripts/train_sft.py --config configs/train_colab.yaml --dry-run

What training teaches (Step 5 rule, Fix 2): examples are built with
``targets.build_chat_messages`` on the shared production prompt, so an empty
dirty field trains keep-empty + ``needs_review`` even though gold holds the
geocoder value. Naive dirty-to-gold pairs would teach invention; this script
refuses to build them (target validation fails fast).

Checkpoint choice (Step 6 rule): the winner is picked on validation repair
metrics — review precision and damage rate alongside F1, never recall alone
(recall alone rewards invention). See ``selection_score``.

Chat template (official MiniCPM recipe): the base model ships an inference
template without ``{% generation %}`` markers, which TRL cannot mask. We set
the training-only template from OpenBMB/MiniCPM ``docs/finetune/trl.md``
(ChatML, no-think, assistant content inside ``{% generation %}``) before
trainer init. It is never saved to disk; inference reloads the original
tokenizer, with which the adapter stays fully compatible per the recipe.
See ``TRAIN_CHAT_TEMPLATE`` and ``apply_training_template``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from addr_repair.io import load_paired_records, records_hash
from addr_repair.prompts import PRODUCTION_PROMPT_REV
from addr_repair.schema import validate_output, validate_output_semantics
from addr_repair.targets import build_chat_messages

# Training-only chat template from the official MiniCPM TRL recipe
# (OpenBMB/MiniCPM docs/finetune/trl.md). The base inference template has no
# {% generation %} markers, so TRL cannot build an assistant-only loss mask
# from it ("not training-compatible" error). This template is ChatML with no
# think blocks; only assistant content falls inside {% generation %}.
# TRAINING-ONLY: never saved to disk, never served. Inference reloads the
# original tokenizer; the adapter stays fully compatible per the recipe.
TRAIN_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}"
    "{{- '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)

TRAIN_TEMPLATE_SOURCE = "OpenBMB/MiniCPM docs/finetune/trl.md (training-only, not saved, not served)"


def apply_training_template(tokenizer):
    """Swap in the training-only template; return the original for the record."""
    original = tokenizer.chat_template
    tokenizer.chat_template = TRAIN_CHAT_TEMPLATE
    return original


def _resolve(path: str | Path, base: Path) -> Path:
    """Resolve a config path: absolute as-is, else beside the current working
    directory (repo root in normal use), else beside the config file.

    Cwd wins on purpose: committed configs are repo-root-relative, and the
    script must behave the same no matter where the (possibly generated)
    train config file lives — e.g. /content/train_run.yaml on Colab must not
    redirect data paths to /content/ when a stray directory exists there.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_hit = Path.cwd() / candidate
    if cwd_hit.exists():
        return cwd_hit
    local = base / candidate
    if local.exists():
        return local
    return cwd_hit


def load_resolved_config(train_config_path: str | Path) -> dict:
    """Load train + model + lora + data configs into one resolved dict."""
    train_path = Path(train_config_path)
    train_cfg = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    root = train_path.parent
    model_cfg = yaml.safe_load(_resolve(train_cfg["base_config"], root).read_text(encoding="utf-8"))
    lora_cfg = yaml.safe_load(_resolve(train_cfg["lora_config"], root).read_text(encoding="utf-8"))
    data_cfg = yaml.safe_load(_resolve(train_cfg["data_config"], root).read_text(encoding="utf-8"))
    return {
        "train_config_path": str(train_path),
        "train": train_cfg,
        "model": model_cfg,
        "lora": lora_cfg,
        "data": data_cfg,
        "root": str(root),
    }


def assert_prompt_parity(resolved: dict) -> None:
    """Fail fast if code prompt rev drifted from the pinned config rev."""
    configured = resolved["model"].get("prompt_rev")
    if configured != PRODUCTION_PROMPT_REV:
        raise SystemExit(
            f"prompt_rev drift: code builds {PRODUCTION_PROMPT_REV!r} but "
            f"configs/model.yaml pins {configured!r}. Bump deliberately, never silently."
        )


def load_split_pairs(resolved: dict, split: str) -> tuple[list[dict], dict]:
    """Load one split's pairs and verify them against the committed manifest."""
    data_cfg = resolved["data"]
    root = Path(resolved["root"])
    manifests_dir = _resolve(data_cfg.get("manifests_dir", "data/manifests"), root)
    raw_dir = _resolve(data_cfg.get("raw_dir", "data/raw"), root)
    manifest_path = manifests_dir / f"{split}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = load_paired_records(
        raw_dir / data_cfg.get("dirty_file", "dirty.csv"),
        raw_dir / data_cfg.get("clean_file", "clean.csv"),
        data_cfg.get("id_field", "id"),
    )
    by_id = {record["id"]: record for record in records}
    wanted = manifest["entity_ids"]
    missing = [entity_id for entity_id in wanted if entity_id not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} manifest IDs missing from raw data ({split})")
    pairs = [by_id[entity_id] for entity_id in wanted]
    actual = records_hash(pairs)
    if actual != manifest["records_sha256"]:
        raise SystemExit(f"{split} records hash differs from manifest; inputs changed")
    return pairs, manifest


def build_message_rows(pairs: list[dict]) -> tuple[list[dict], dict]:
    """Build conversational SFT rows; fail fast if any target breaks contract."""
    rows = []
    flagged = 0
    changed = 0
    invalid: list[str] = []
    for pair in pairs:
        dirty, gold = pair["dirty"], pair["gold"]
        messages = build_chat_messages(dirty, gold)
        payload = json.loads(messages[-1]["content"])
        ok, _ = validate_output(payload)
        errors = validate_output_semantics(dirty, payload)
        if not ok or errors:
            invalid.append(pair["id"])
            continue
        rows.append({"messages": messages})
        flagged += len(payload["needs_review"])
        changed += len(payload["changes"])
    if invalid:
        raise SystemExit(
            f"{len(invalid)} training targets fail validation "
            f"(e.g. {invalid[0]}); refusing to teach invalid outputs"
        )
    stats = {
        "n": len(rows),
        "targets_with_changes": sum(1 for r in rows if json.loads(r["messages"][-1]["content"])["changes"]),
        "review_flags": flagged,
        "assistant_fields_changed": changed,
        "invalid_targets": 0,
    }
    return rows, stats


def selection_score(metrics: dict) -> float:
    """Val selection composite: honesty first, never recall alone.

    Mean of review precision, (1 - damage rate), and repair F1. A checkpoint
    that invents its way to high recall scores review precision near zero and
    loses to an honest abstainer — which is the intended Step 6 behaviour.
    """
    return round(
        (metrics["review_precision"] + (1.0 - metrics["damage_rate"]) + metrics["repair_f1"]) / 3.0,
        4,
    )


def pick_best(scored: list[tuple[str, dict]]) -> tuple[str, dict]:
    """Pick the val-best checkpoint. Ties: higher contract, fewer inventions."""

    def _key(item: tuple[str, dict]) -> tuple:
        _name, metrics = item
        return (
            selection_score(metrics),
            metrics.get("contract_validity", 0.0),
            -metrics.get("inventions", 0),
        )

    return max(scored, key=_key)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    """Return the newest checkpoint dir, or None when starting fresh."""
    candidates = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _empty_cuda_cache() -> None:
    if importlib.util.find_spec("torch") is None:
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _repo_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_record() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "cuda_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "torch_version": torch.__version__,
            }
        return {"cuda_available": False, "torch_version": torch.__version__}
    except (ImportError, RuntimeError) as exc:
        return {"cuda_available": False, "error": str(exc)}


def run_dry_run(resolved: dict) -> int:
    """Log the full training plan without GPU or downloads. Returns exit code."""
    assert_prompt_parity(resolved)
    train_pairs, train_manifest = load_split_pairs(resolved, "train")
    val_pairs, val_manifest = load_split_pairs(resolved, "val")
    _, train_stats = build_message_rows(train_pairs)
    _, _val_stats = build_message_rows(val_pairs)

    train_cfg = resolved["train"]
    batch = train_cfg["per_device_train_batch_size"] * train_cfg["gradient_accumulation_steps"]
    steps_per_epoch = max(1, (len(train_pairs) + batch - 1) // batch)
    total_steps = steps_per_epoch * train_cfg["num_train_epochs"]
    print("[train_sft] DRY RUN — no GPU, no downloads, no writes")
    print(f"[train_sft] model={resolved['model']['model_id']} rev={resolved['model']['model_rev']}")
    print(f"[train_sft] prompt_rev={resolved['model']['prompt_rev']} (code {PRODUCTION_PROMPT_REV})")
    print(f"[train_sft] chat_template_mode={resolved['model'].get('chat_template_mode')}")
    print(
        f"[train_sft] training template: official MiniCPM recipe, "
        f"sha={hashlib.sha256(TRAIN_CHAT_TEMPLATE.encode()).hexdigest()[:16]}, "
        "gen-markers=yes, think-blocks=no (training-only, never saved/served)"
    )
    print(
        f"[train_sft] train n={len(train_pairs)} hash={train_manifest['records_sha256'][:16]} "
        f"val n={len(val_pairs)} hash={val_manifest['records_sha256'][:16]}"
    )
    print(
        f"[train_sft] targets valid: train invalid=0 "
        f"(changes rows={train_stats['targets_with_changes']} review flags={train_stats['review_flags']})"
    )
    print(
        f"[train_sft] lora r={resolved['lora']['r']} alpha={resolved['lora']['lora_alpha']} "
        f"assistant_only_loss={resolved['lora'].get('assistant_only_loss')}"
    )
    print(
        f"[train_sft] epochs={train_cfg['num_train_epochs']} ~{total_steps} steps "
        f"(~{steps_per_epoch}/epoch, batch {batch} = "
        f"{train_cfg['per_device_train_batch_size']}x{train_cfg['gradient_accumulation_steps']})"
    )
    print(
        f"[train_sft] eval_steps={train_cfg.get('eval_steps')} save_steps={train_cfg.get('save_steps')} "
        f"save_total_limit={train_cfg.get('save_total_limit')}"
    )
    print(
        f"[train_sft] val selection: {train_cfg.get('val_samples', 200)} samples, "
        "winner = max mean(review_precision, 1-damage, F1), ties by contract then fewest inventions"
    )
    print(f"[train_sft] output_dir={train_cfg['output_dir']}")
    print("[train_sft] parity: prompt text shared with inference (build_repair_prompt); "
          "training template = official recipe (recorded hash); "
          "inference reloads original tokenizer per recipe")
    return 0


def build_sft_config(train_cfg: dict, output_dir: Path, use_bf16: bool):
    """Build the TRL SFTConfig, dropping (and reporting) unsupported kwargs.

    TRL/transformers field names drift across versions (e.g. warmup_ratio
    missing in transformers 5.x) and the dry-run never constructs this
    object — so a typo or drifted name would only explode on the paid GPU.
    This helper is covered by a test that really constructs SFTConfig.
    Returns (sft_args, dropped_kwargs).
    """
    from trl import SFTConfig

    wanted = {
        "output_dir": str(output_dir),
        "num_train_epochs": train_cfg["num_train_epochs"],
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "lr_scheduler_type": train_cfg.get("lr_scheduler_type", "cosine"),
        "warmup_steps": train_cfg.get("warmup_steps", 30),
        "bf16": use_bf16,
        "fp16": bool(train_cfg.get("fp16", False)),
        "max_length": train_cfg.get("max_seq_length", 2048),
        "packing": False,
        "eval_strategy": train_cfg.get("eval_strategy", "steps"),
        "eval_steps": train_cfg.get("eval_steps", 200),
        "save_steps": train_cfg.get("save_steps", 200),
        "save_total_limit": train_cfg.get("save_total_limit", 3),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": train_cfg.get("logging_steps", 50),
        "report_to": train_cfg.get("report_to", "none"),
        "seed": train_cfg.get("seed", 7),
        "remove_unused_columns": False,
        "assistant_only_loss": True,
    }
    supported = set(SFTConfig.__dataclass_fields__)
    dropped = sorted(k for k in wanted if k not in supported)
    kept = {k: v for k, v in wanted.items() if k in supported}
    return SFTConfig(**kept), dropped


def score_val_sample(    model,
    tokenizer,
    pairs: list[dict],
    *,
    max_samples: int = 200,
    max_new_tokens: int = 512,
) -> dict:
    """Score one model on a deterministic val subsample (greedy, HF pipeline).

    Comparative signal for checkpoint choice only — final evidence always comes
    from the frozen GGUF eval. Lazy heavy imports live with the caller.
    """
    from addr_repair.parsing import parse_response
    from addr_repair.prompts import build_repair_prompt
    from addr_repair.schema import validate_output, validate_output_semantics
    from addr_repair.scorer import score_records

    sample = pairs[:max(1, max_samples)]
    rows = []
    for pair in sample:
        dirty = pair["dirty"]
        messages = build_repair_prompt(dirty)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        payload, _parse_errors = parse_response(decoded)
        if payload is None:
            rows.append(
                {
                    "dirty": dirty, "gold": pair["gold"], "pred": dict(dirty),
                    "schema_ok": False, "semantic_ok": False, "parsed": False,
                    "needs_review": [],
                }
            )
            continue
        ok, _ = validate_output(payload)
        semantic_errors = validate_output_semantics(dirty, payload)
        rows.append(
            {
                "dirty": dirty, "gold": pair["gold"], "pred": payload["clean_record"],
                "schema_ok": ok, "semantic_ok": not semantic_errors, "parsed": True,
                "needs_review": payload.get("needs_review", []),
            }
        )
    return score_records(rows)


def run_training(resolved: dict, resume: bool = False) -> dict:
    """Full SFT run: TRL + LoRA, val selection, hashes + GPU/time record.

    ``resume=True`` continues from the latest checkpoint in ``output_dir``
    (Colab disconnect recovery); without any checkpoint it starts fresh.
    """
    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTTrainer

    assert_prompt_parity(resolved)
    train_cfg, model_cfg, lora_cfg = resolved["train"], resolved["model"], resolved["lora"]
    started = time.perf_counter()
    transformers.set_seed(train_cfg.get("seed", 7))

    train_pairs, train_manifest = load_split_pairs(resolved, "train")
    val_pairs, val_manifest = load_split_pairs(resolved, "val")
    train_rows, train_stats = build_message_rows(train_pairs)
    val_rows, val_stats = build_message_rows(val_pairs)

    model_id = model_cfg["model_id"]
    model_rev = model_cfg["model_rev"]
    # Standard LlamaForCausalLM per the official recipe: no remote code needed.
    trust_remote = bool(train_cfg.get("trust_remote_code", False))
    use_bf16 = bool(train_cfg.get("bf16", True))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, revision=model_rev, trust_remote_code=trust_remote, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    template_before = apply_training_template(tokenizer)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_rev,
        trust_remote_code=trust_remote,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    output_dir = Path(train_cfg["output_dir"])
    sft_args, dropped_sft_kwargs = build_sft_config(train_cfg, output_dir, use_bf16)
    if dropped_sft_kwargs:
        print(f"[train_sft] WARNING: SFTConfig ignores unsupported args: {dropped_sft_kwargs}")
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    template_swapped = trainer.chat_template is not None
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {"train": train_cfg, "model": model_cfg, "lora": lora_cfg, "data": resolved["data"]},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    resume_from = output_dir if (resume and _latest_checkpoint(output_dir)) else None
    if resume and resume_from is None:
        print("[train_sft] --resume set but no checkpoint found; starting fresh")
    resumed = resume_from is not None
    trainer.train(resume_from_checkpoint=resume_from)
    train_seconds = round(time.perf_counter() - started, 1)

    # Free the training model before val scoring loads fresh copies.
    del trainer
    del model
    gc.collect()
    _empty_cuda_cache()

    # Step 6 selection: score retained checkpoints + final adapter on val sample.
    val_samples = int(train_cfg.get("val_samples", 200))
    val_gen_tokens = int(train_cfg.get("val_gen_max_tokens", 512))
    candidates = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    scored: list[tuple[str, dict]] = []
    val_started = time.perf_counter()
    base_model_id, base_rev = model_id, model_rev
    for checkpoint in candidates + [output_dir]:
        adapter = checkpoint / "adapter_model.safetensors"
        if not adapter.is_file():
            adapter = checkpoint / "adapter_model.bin"
        if not adapter.is_file() and checkpoint != output_dir:
            continue
        from peft import PeftModel

        plain = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_id, revision=base_rev, trust_remote_code=trust_remote,
            dtype=torch.bfloat16 if use_bf16 else torch.float32, device_map="auto",
        )
        scored_model = PeftModel.from_pretrained(plain, str(checkpoint)) if adapter.is_file() else plain
        scored_model.eval()
        metrics = score_val_sample(
            scored_model, tokenizer, val_pairs,
            max_samples=val_samples, max_new_tokens=val_gen_tokens,
        )
        metrics["adapter_sha256"] = _sha256_file(adapter) if adapter.is_file() else None
        scored.append((checkpoint.name, metrics))
        del scored_model, plain
        _empty_cuda_cache()
    val_seconds = round(time.perf_counter() - val_started, 1)
    winner_name, winner_metrics = pick_best(scored)

    record = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "resumed": resumed,
        "model_id": model_id,
        "model_rev": model_rev,
        "prompt_rev": model_cfg.get("prompt_rev"),
        "chat_template_mode": model_cfg.get("chat_template_mode"),
        "training_template_sha256": hashlib.sha256(
            TRAIN_CHAT_TEMPLATE.encode("utf-8")
        ).hexdigest(),
        "training_template_source": TRAIN_TEMPLATE_SOURCE,
        "inference_template_sha256": hashlib.sha256(
            (template_before or "").encode("utf-8")
        ).hexdigest(),
        "trl_swapped_training_template": template_swapped,
        "lora": {k: lora_cfg.get(k) for k in ("r", "lora_alpha", "lora_dropout", "target_modules")},
        "assistant_only_loss": True,
        "dropped_sft_kwargs": dropped_sft_kwargs,
        "train_manifest_sha256": train_manifest["records_sha256"],
        "val_manifest_sha256": val_manifest["records_sha256"],
        "train_target_stats": train_stats,
        "val_target_stats": val_stats,
        "train_seconds": train_seconds,
        "val_scoring_seconds": val_seconds,
        "checkpoints": [
            {"name": name, "selection_score": selection_score(m), "metrics": m}
            for name, m in scored
        ],
        "winner": winner_name,
        "winner_selection_score": selection_score(winner_metrics),
        "selection_rule": "max mean(review_precision, 1-damage_rate, repair_f1); ties by contract_validity then fewest inventions",
        "gpu": _gpu_record(),
        "platform": f"{platform.system()} {platform.machine()}",
        "repo_commit": _repo_commit(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    try:
        import peft as _peft
        import trl as _trl

        record["versions"]["trl"] = _trl.__version__
        record["versions"]["peft"] = _peft.__version__
    except ImportError:
        pass
    (output_dir / "training_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "selection.json").write_text(
        json.dumps(
            {"winner": winner_name, "selection_score": selection_score(winner_metrics),
             "metrics": winner_metrics, "rule": record["selection_rule"]},
            ensure_ascii=False, indent=2, sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[train_sft] done in {train_seconds}s; winner={winner_name} "
          f"score={selection_score(winner_metrics)} -> {output_dir}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA SFT for MiniCPM5-1B address repair.")
    parser.add_argument("--config", default="configs/train_colab.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Log plan without training.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue from the latest checkpoint in output_dir (Colab recovery).",
    )
    args = parser.parse_args()
    resolved = load_resolved_config(args.config)
    if args.dry_run:
        raise SystemExit(run_dry_run(resolved))
    run_training(resolved, resume=args.resume)


if __name__ == "__main__":
    main()
