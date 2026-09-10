"""Frozen evaluation runner. Rules, base, and SFT score through one path.

Usage:
    uv run python scripts/evaluate_local.py --system rules
    uv run python scripts/evaluate_local.py --system base --manifest data/manifests/test.json
    uv run python scripts/evaluate_local.py --system sft --gguf models/sft-Q4_K_M.gguf --model-rev <adapter-hash>

The runner checks the manifest hash before scoring, writes one audit record
per input, and saves metrics plus a freeze record (hashes only, no raw
records). Start the llama.cpp server first for base/sft:
llama-server -m <gguf> -c 2048 --reasoning off

FREEZE: do not change prompt, normalizer, model, or test data after this.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from addr_repair.audit import write_audit_record
from addr_repair.inference import LocalModel, run_rules
from addr_repair.io import load_paired_records, records_hash, sha256_file
from addr_repair.prompts import PRODUCTION_PROMPT_REV
from addr_repair.schema import validate_output, validate_output_semantics
from addr_repair.scorer import score_records


def _repo_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _load_test_pairs(manifest_path: Path, raw_dir: Path, id_field: str) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = load_paired_records(raw_dir / "dirty.csv", raw_dir / "clean.csv", id_field)
    by_id = {record["id"]: record for record in records}
    wanted = manifest["entity_ids"]
    missing = [entity_id for entity_id in wanted if entity_id not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} manifest IDs missing from raw data")
    pairs = [by_id[entity_id] for entity_id in wanted]
    actual = records_hash(pairs)
    if actual != manifest["records_sha256"]:
        raise SystemExit("test records hash differs from manifest; inputs changed after freezing")
    return pairs, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default="rules", choices=["rules", "base", "sft"])
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--manifest", default="data/manifests/test.json")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--out-dir", default="evals/frozen-test/")
    parser.add_argument("--gguf", default=None, help="local GGUF path (defaults to model.yaml local_gguf)")
    parser.add_argument("--model-rev", default=None, help="required for sft: adapter/checkpoint hash")
    args = parser.parse_args()

    model_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if model_config.get("prompt_rev") != PRODUCTION_PROMPT_REV:
        raise SystemExit(
            f"prompt_rev drift: code serves {PRODUCTION_PROMPT_REV!r} but "
            f"{args.config} pins {model_config.get('prompt_rev')!r}"
        )
    pairs, manifest = _load_test_pairs(Path(args.manifest), Path(args.raw_dir), args.id_field)

    gguf_path = Path(args.gguf) if args.gguf else Path(model_config["local_gguf"])
    if args.system == "rules":
        model_rev, run = "rules-floor-v1", run_rules
        gguf_sha = None
    else:
        if args.system == "sft" and not args.model_rev:
            raise SystemExit("sft needs --model-rev (adapter/checkpoint hash)")
        if not gguf_path.is_file():
            raise SystemExit(f"GGUF not found: {gguf_path}")
        model_rev = args.model_rev or model_config["model_rev"]
        client = LocalModel(
            server_url=model_config["server_url"],
            model=gguf_path.name,
            model_rev=model_rev,
            decode=model_config["decode"],
        )
        run = client.repair
        gguf_sha = sha256_file(gguf_path)

    system_dir = Path(args.out_dir) / args.system
    audit_path = system_dir / "audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    score_rows = []
    latencies = []
    for pair in pairs:
        dirty = pair["dirty"]
        output, meta = run(dirty)
        latencies.append(meta["latency_ms"])
        if output is None:
            ok, semantic_errors = False, meta["parse_errors"]
            parsed, semantic_ok = False, False
            audit_output: dict = {"raw_text": meta["raw_text"], "parse_errors": meta["parse_errors"]}
            pred = dict(dirty)  # unusable output counts as no repair
        else:
            ok, _ = validate_output(output)
            semantic_errors = validate_output_semantics(dirty, output)
            parsed, semantic_ok = True, not semantic_errors
            audit_output = output
            pred = output["clean_record"]
        write_audit_record(
            audit_path, dirty, audit_output,
            model_rev=meta["model_rev"], prompt_rev=meta["prompt_rev"],
            schema_ok=ok, latency_ms=meta["latency_ms"],
            semantic_errors=semantic_errors,
        )
        score_rows.append(
            {
                "dirty": dirty,
                "gold": pair["gold"],
                "pred": pred,
                "schema_ok": ok,
                "semantic_ok": semantic_ok,
                "parsed": parsed,
                "needs_review": audit_output.get("needs_review", []),
            }
        )

    metrics = score_records(score_rows)
    metrics["latency_ms_median"] = round(statistics.median(latencies), 1)
    metrics["latency_ms_p95"] = round(
        statistics.quantiles(latencies, n=100)[94] if len(latencies) > 1 else latencies[0], 1
    )
    metrics["latency_ms_n"] = len(latencies)
    freeze = {
        "system": args.system,
        "model_rev": model_rev,
        "gguf": str(gguf_path),
        "gguf_sha256": gguf_sha,
        "manifest": str(args.manifest),
        "manifest_records_sha256": manifest["records_sha256"],
        "source_dirty_sha256": manifest.get("dirty_sha256"),
        "source_clean_sha256": manifest.get("clean_sha256"),
        "prompt_rev": model_config.get("prompt_rev"),
        "decode": model_config.get("decode"),
        "n_ctx": model_config.get("n_ctx"),
        "llama_cpp_rev": model_config.get("llama_cpp_rev"),
        "repo_commit": _repo_commit(),
        "platform": f"{platform.system()} {platform.machine()}",
    }
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "metrics.json").write_text(
        json.dumps({"freeze": freeze, "metrics": metrics}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (Path(args.out_dir) / "freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[evaluate_local] system={args.system} n={len(pairs)} -> {system_dir}")
    print(
        f"[evaluate_local] precision={metrics['repair_precision']} "
        f"recall={metrics['repair_recall']} f1={metrics['repair_f1']} "
        f"damage={metrics['damage_rate']} schema={metrics['schema_validity']} "
        f"semantic={metrics['semantic_validity']} contract={metrics['contract_validity']} "
        f"inventions={metrics['inventions']} "
        f"latency_p50={metrics['latency_ms_median']}ms p95={metrics['latency_ms_p95']}ms"
    )


if __name__ == "__main__":
    main()
