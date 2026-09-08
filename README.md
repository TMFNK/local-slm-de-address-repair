# local-slm-de-address-repair

Fine-tune `openbmb/MiniCPM5-1B` to repair messy German address records into validated JSON, run it locally on an M2 Air (8 GB), and show where it repairs correctly, damages clean fields, or asks for human review.

v1 scope: LoRA supervised fine-tuning only. No GRPO. No invented missing data, no geocoding, no entity matching, no fraud work.

## Planned workflow

```bash
uv sync
uv run python scripts/prepare_data.py --config configs/data.yaml
uv run python scripts/run_baseline.py --config configs/model.yaml  # rules floor
uv run python scripts/train_sft.py --config configs/train_colab.yaml   # Colab GPU
uv run python scripts/evaluate_local.py --config configs/model.yaml    # M2 Air, llama.cpp Q4
uv run python scripts/make_report.py --evals evals/frozen-test/
```

Full steps: `docs/REPRO.md`. What we measure: `docs/EVAL.md`. Splits: `docs/SPLITS.md`.

The repository is being implemented in evaluation-first steps. Step 1 is
complete: the rules-floor path, field-level scorer, JSON Schema validation,
semantic output validation, and audit fields are implemented and tested.
Dataset preparation, MiniCPM5 inference, SFT training, and frozen local
evaluation remain scaffold work until their commands are implemented.

## Data: do not commit raw records

Raw `dirty.csv` / `clean.csv` never go into git. Download the published Zenodo release of [Clean Me If You Can](https://github.com/D2IP-TUB/Clean-Me-If-You-Can) during setup, record checksum + URL + date in your split manifest.

- Source data: OpenStreetMap contributors, ODbL 1.0.
- This repo keeps only split manifests (hashes), a 100-record smoke fixture, code, configs, and metrics.
- `data/` is gitignored. See `.gitignore`.

## Systems compared (same frozen test set)

1. **Rules floor** — deterministic whitespace, casing, Unicode, abbreviation normalizer.
2. **Base** — untuned MiniCPM5-1B + production prompt + schema check + review policy.
3. **SFT** — same model after LoRA fine-tuning, checkpoint picked on validation only.

## Output contract and audit

Every response must validate against the fixed JSON Schema in
`src/addr_repair/schema.py`: `clean_record` + `changes` + `needs_review`.
Semantic validation additionally checks that changes match the dirty input and
returned clean record, reviewed fields remain unchanged, and every changed
field has an audit entry.

Every run appends one JSONL audit record per input containing the input hash,
model and prompt revisions, output, schema result, semantic result, latency,
and score when gold exists.

Run the implemented test and lint checks with:

```bash
uv run pytest
uv run ruff check src scripts tests
```

## Hardware

- Training: Colab GPU (see `notebooks/colab_sft.ipynb`, thin wrapper around `scripts/train_sft.py`).
- Eval + demo: M2 Air 8 GB via `llama.cpp`, Q4 quantization, same prompt and decoder for base and tuned.

## License and scope note

Code: Apache-2.0 (`LICENSE`). Data terms: ODbL 1.0 for the source address data. This work evaluates public address records, not private customer data.
