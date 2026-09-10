# Reproduction

## Current implementation

The repository is built in evaluation-first steps. Step 1 is runnable:

```bash
uv sync
uv run pytest
uv run ruff check src scripts tests
```

Data preparation is implemented but requires the pinned source release in
`data/raw/dirty.csv` and `data/raw/clean.csv`. After recording the release
metadata and checksums in `configs/data.yaml`, run:

```bash
uv run python scripts/prepare_data.py --config configs/data.yaml
```

The command refuses incomplete pairs, duplicate IDs, insufficient entity
counts, or accidental replacement of generated manifests. It writes one
manifest per split and a paired 100-record fixture from the held-out test
partition. Pass `--force` only when deliberately regenerating those files.

The rules-floor smoke runner is the first experiment gate:

```bash
uv run python scripts/run_baseline.py --config configs/model.yaml
```

With gold pairs it also writes scored metrics next to the audit output:

```bash
uv run python scripts/run_baseline.py --pairs fixtures/smoke_100/pairs.jsonl \
  --out evals/baseline-smoke/audit.jsonl
```

This writes `evals/baseline-smoke/metrics.json` (aggregate plus per-field
repair precision/recall, damage rate, and review precision) alongside the
ignored audit JSONL. `--pairs` and `--fixture` are mutually exclusive.

Frozen evaluation runs every system through one shared local path
(parse, validate, audit, score). Start the pinned server first; the
`--reasoning off` flag is what engages no-think mode:

```bash
./llama.cpp/build/bin/llama-server -m models/MiniCPM5-1B-Q4_K_M.gguf -c 2048 --reasoning off
uv run python scripts/evaluate_local.py --system rules
uv run python scripts/evaluate_local.py --system base --manifest fixtures/smoke_100/manifest.json --out-dir evals/baseline-smoke/
```

Each run checks the manifest hash before scoring and writes ignored
`audit.jsonl` plus committed `metrics.json` and `freeze.json` (hashes and
aggregate numbers only, no raw records). SFT needs its adapter GGUF and
checkpoint hash:

```bash
uv run python scripts/evaluate_local.py --system sft --gguf models/sft-Q4_K_M.gguf --model-rev <adapter-hash>
```

## Planned experiment

1. Download the Zenodo release from `configs/data.yaml`; record URL, date,
   and checksum.
2. Run `scripts/prepare_data.py` to create manifests, splits, and the smoke
   fixture.
3. Run the rules floor and base-model smoke evaluation.
4. Open `notebooks/colab_sft.ipynb`; it calls `scripts/train_sft.py`
   (implemented: TRL LoRA SFT on evidence-preserving targets, assistant-only
   loss, honesty-based checkpoint selection; see the vault Fix 1–6 notes).
   Verify first with `--dry-run`, which needs no GPU.
5. Select the checkpoint using validation data only and save checkpoint hashes.
6. Merge the adapter into the base model, then convert the merged checkpoint
   to GGUF using a pinned `llama.cpp` revision.
7. Run the same frozen test set through rules, base, and SFT systems.
8. Generate the report with raw-output and configuration hashes.

`train_sft.py` is implemented; `make_report.py` is still a TODO placeholder.
Do not treat a printed placeholder as an experiment result.

## Pin record

Record Python, TRL, Transformers, PEFT, `llama.cpp` revision, MiniCPM5
revision, dataset release, prompt revision, decoder settings, and checkpoint
hashes in configs, manifests, and `uv.lock`.
