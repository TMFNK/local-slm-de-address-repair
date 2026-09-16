"""Step 4 offline reward check (plan 2026-09-15).

Samples the frozen v3 GGUF on 100 val records x 4 answers and scores each
answer with rewards.py. Batched and resumable for a small Mac.

Sampling (one batch)::

    uv run python scripts/reward_check.py --batch-start 0 --batch-count 10

Analysis (after all 100 records are sampled)::

    uv run python scripts/reward_check.py --analyze

Outputs (new dir; frozen weights, lists, and past evals/ files untouched):
``evals/reward-check-v1/samples.jsonl`` and ``evals/reward-check-v1/analysis.json``.

Start the server first:
llama-server -m models/sft-Q4_K_M-clean-gold-v3.gguf -c 2048 --reasoning off
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from addr_repair.inference import LocalModel
from addr_repair.io import load_paired_records, records_hash
from addr_repair.rewards import score_answer
from addr_repair.schema import validate_output, validate_output_semantics
from addr_repair.scorer import score_records

VAL_N = 100
SAMPLES = 4
SAMPLING_TEMP = 0.7
SEED_BASE = 1000


def _load_val_pairs(manifest_path: Path, raw_dir: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = load_paired_records(raw_dir / "dirty.csv", raw_dir / "clean.csv", "id")
    by_id = {record["id"]: record for record in records}
    wanted = manifest["entity_ids"][:VAL_N]
    missing = [entity_id for entity_id in wanted if entity_id not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} manifest IDs missing from raw data")
    pairs = [by_id[entity_id] for entity_id in wanted]
    if manifest.get("records_sha256"):
        full = [by_id[entity_id] for entity_id in manifest["entity_ids"]]
        if records_hash(full) != manifest["records_sha256"]:
            raise SystemExit("val records hash differs from manifest")
    return pairs, manifest


def _read_done(samples_path: Path) -> set[tuple[int, int]]:
    done = set()
    if samples_path.is_file():
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["idx"], row["sample"]))
    return done


def _sample_batch(args: argparse.Namespace) -> None:
    pairs, _manifest = _load_val_pairs(Path(args.manifest), Path(args.raw_dir))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    done = _read_done(samples_path)
    selected = pairs[args.batch_start : args.batch_start + args.batch_count]
    print(
        f"[reward_check] val ids {args.batch_start}.."
        f"{args.batch_start + len(selected) - 1}, "
        f"{len(done)} samples already stored"
    )
    with samples_path.open("a", encoding="utf-8") as fh:
        for offset, pair in enumerate(selected):
            idx = args.batch_start + offset
            for sample in range(SAMPLES):
                if (idx, sample) in done:
                    continue
                seed = SEED_BASE + idx * SAMPLES + sample
                client = LocalModel(
                    server_url=args.server_url,
                    model=Path(args.gguf).name,
                    model_rev=args.model_rev,
                    decode={
                        "temperature": SAMPLING_TEMP,
                        "top_p": 1.0,
                        "max_tokens": 512,
                        "seed": seed,
                    },
                )
                output, meta = client.repair(pair["dirty"])
                reward = score_answer(pair["dirty"], pair["gold"], meta["raw_text"])
                fh.write(
                    json.dumps(
                        {
                            "idx": idx,
                            "entity_id": pair["id"],
                            "sample": sample,
                            "seed": seed,
                            "temperature": SAMPLING_TEMP,
                            "raw_text": meta["raw_text"],
                            "output": output,
                            "latency_ms": meta["latency_ms"],
                            "finish_reason": meta.get("finish_reason"),
                            "reward": reward,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                fh.flush()
                print(
                    f"[reward_check] idx={idx} sample={sample} "
                    f"total={reward['total']} structure={reward['structure_total']} "
                    f"parsed={reward['parsed_ok']} {meta['latency_ms']}ms"
                )
    print(f"[reward_check] batch stored -> {samples_path}")


def _scorer_rows(pairs: list[dict], picks: dict[int, dict]) -> list[dict]:
    rows = []
    for idx, pair in enumerate(pairs):
        output = picks[idx]["output"]
        if output is None:
            rows.append(
                {
                    "dirty": pair["dirty"],
                    "gold": pair["gold"],
                    "pred": dict(pair["dirty"]),
                    "schema_ok": False,
                    "semantic_ok": False,
                    "parsed": False,
                    "needs_review": [],
                }
            )
        else:
            ok, _ = validate_output(output)
            errors = validate_output_semantics(pair["dirty"], output)
            rows.append(
                {
                    "dirty": pair["dirty"],
                    "gold": pair["gold"],
                    "pred": output["clean_record"],
                    "schema_ok": ok,
                    "semantic_ok": not errors,
                    "parsed": True,
                    "needs_review": output.get("needs_review", []),
                }
            )
    return rows


def _analyze(args: argparse.Namespace) -> None:
    pairs, _manifest = _load_val_pairs(Path(args.manifest), Path(args.raw_dir))
    rows = [
        json.loads(line)
        for line in (Path(args.out_dir) / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_idx: dict[int, list[dict]] = {}
    for row in rows:
        by_idx.setdefault(row["idx"], []).append(row)
    if len(by_idx) < VAL_N or any(len(v) != SAMPLES for v in by_idx.values()):
        have = sum(len(v) for v in by_idx.values())
        raise SystemExit(f"need {VAL_N}x{SAMPLES}=400 samples, have {have}; sample more batches")
    for group in by_idx.values():
        group.sort(key=lambda r: r["sample"])

    totals = [r["reward"]["total"] for r in rows]
    spread = {
        "mean": round(statistics.mean(totals), 4),
        "stdev": round(statistics.pstdev(totals), 4),
        "min": min(totals),
        "max": max(totals),
        "groups_with_spread": sum(
            len({r["reward"]["total"] for r in group}) > 1 for group in by_idx.values()
        ),
    }

    def pick(key: str) -> dict[int, dict]:
        return {
            idx: max(group, key=lambda r: (r["reward"][key], -r["sample"]))
            for idx, group in by_idx.items()
        }

    single = {idx: group[0] for idx, group in by_idx.items()}
    best_full = pick("total")
    best_shape = pick("structure_total")

    single_m = score_records(_scorer_rows(pairs, single))
    full_m = score_records(_scorer_rows(pairs, best_full))
    shape_m = score_records(_scorer_rows(pairs, best_shape))

    def fills_adds(picks: dict[int, dict]) -> dict:
        fills = sum(r["reward"]["n_invention"] for r in picks.values())
        adds = sum(r["reward"]["n_addition"] for r in picks.values())
        flagged = sum(len((r["output"] or {}).get("needs_review", [])) for r in picks.values())
        return {"fills": fills, "additions": adds, "review_flags": flagged}

    report = {
        "n_records": VAL_N,
        "samples_per_record": SAMPLES,
        "temperature": SAMPLING_TEMP,
        "manifest": str(args.manifest),
        "gguf": str(args.gguf),
        "model_rev": args.model_rev,
        "spread": spread,
        "single_answer": {
            "f1": single_m["repair_f1"],
            "precision": single_m["repair_precision"],
            "recall": single_m["repair_recall"],
            **fills_adds(single),
        },
        "best_full_reward": {
            "f1": full_m["repair_f1"],
            "precision": full_m["repair_precision"],
            "recall": full_m["repair_recall"],
            "damage": full_m["damage_rate"],
            **fills_adds(best_full),
        },
        "best_shape_only": {
            "f1": shape_m["repair_f1"],
            "precision": shape_m["repair_precision"],
            "recall": shape_m["repair_recall"],
            "damage": shape_m["damage_rate"],
            **fills_adds(best_shape),
        },
    }
    out = Path(args.out_dir) / "analysis.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifests/val.json")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--gguf", default="models/sft-Q4_K_M-clean-gold-v3.gguf")
    parser.add_argument(
        "--model-rev",
        default="7103451b9cc916920c59e5d68e8c3ada852b294be5eab55a0d1a9abdc77ec712",
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--out-dir", default="evals/reward-check-v1")
    parser.add_argument("--batch-start", type=int, default=0)
    parser.add_argument("--batch-count", type=int, default=10)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        _analyze(args)
    else:
        _sample_batch(args)


if __name__ == "__main__":
    main()
