# local-slm-de-address-repair

Fine-tune `openbmb/MiniCPM5-1B` to repair messy German address records into validated JSON, and measure where it repairs correctly, damages clean fields, or asks for human review.

Version 1 uses LoRA supervised fine-tuning. It does not invent missing address data, geocode records, or perform entity matching.

## Project status

Step 1 is complete: the rules floor, field-level scorer, JSON Schema
validation, semantic output validation, and audit fields are implemented and
tested. Dataset preparation, model inference, SFT training, and frozen
evaluation remain scaffold work.

## Task

Input: one dirty German named-address record.

Output: a normalized record, field-level change list, and `needs_review`
fields where the input does not provide enough evidence.

The system compares a deterministic rules floor, the untuned base model, and
the SFT model on the same held-out records.

## Quick start

```bash
uv sync
uv run pytest
uv run ruff check src scripts tests
```

The planned experiment workflow is documented in `docs/REPRO.md`. The
preparation, training, evaluation, and report commands will become runnable as
the remaining scaffold steps are implemented.

## Evaluation design

1. **Rules floor** — deterministic whitespace, casing, Unicode, abbreviation normalizer.
2. **Base** — untuned MiniCPM5-1B + production prompt + schema check + review policy.
3. **SFT** — same model after LoRA fine-tuning, checkpoint picked on validation only.

Metrics are calculated against paired dirty and clean records:

- repair precision, recall, and F1;
- clean-field damage rate;
- correct abstention and review precision;
- schema and semantic validity;
- per-field error counts;
- latency and model resource measurements once the runtime is implemented.

See `docs/EVAL.md` for the scoring definitions.

## Output contract and audit

Every response must validate against the fixed JSON Schema in
`src/addr_repair/schema.py`: `clean_record` + `changes` + `needs_review`.
Semantic validation additionally checks that changes match the dirty input and
returned clean record, reviewed fields remain unchanged, and every changed
field has an audit entry.

Every run appends one JSONL audit record per input containing the input hash,
model and prompt revisions, output, schema result, semantic result, latency,
and score when gold exists.

## Reproduction and data

Training uses a GPU through the thin runner in
`notebooks/colab_sft.ipynb`. Local serving and evaluation use `llama.cpp`.
The base and tuned models must use the same prompt, decoder settings, runtime
revision, and quantization procedure.

Raw source records never go into git. Download the published
[Clean Me If You Can](https://github.com/D2IP-TUB/Clean-Me-If-You-Can) release,
record its URL, date, and checksum, and keep only manifests, fixtures, code,
configs, and result metadata in this repository.

See:

- `docs/REPRO.md` — reproduction workflow and pinned inputs
- `docs/SPLITS.md` — entity-disjoint data split
- `docs/EVAL.md` — scoring and validation
- `paper/outline.md` — technical report structure

Code: Apache-2.0 (`LICENSE`). Source address data: ODbL 1.0.
