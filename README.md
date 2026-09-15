# MiniCPM5 German Address Repair

Fine-tune a small language model to repair messy German named-address records
without silently inventing information.

The project tests whether `openbmb/MiniCPM5-1B` can handle cases that
deterministic normalization does not cover: inconsistent spacing, casing,
Unicode, punctuation, documented abbreviations, obvious spelling defects, and
misplaced address fields.

The result is a local data-cleaning pipeline with an explicit audit trail:
every changed field is recorded, unsupported repairs are routed to
`needs_review`, and clean fields are measured for accidental damage.

Keywords: `MiniCPM5`, `small language model`, `data cleaning`, `address
normalization`, `German addresses`, `LoRA`, `local inference`, `reproducible
evaluation`.

## What it does

Input:

```json
{
  "name": "  Example GmbH ",
  "road": "Musterstr.",
  "house_number": "12",
  "postcode": "80331",
  "locality": "Muenchen",
  "country_code": "de"
}
```

Output:

```json
{
  "clean_record": {
    "name": "Example GmbH",
    "road": "Musterstraße",
    "house_number": "12",
    "postcode": "80331",
    "locality": "Muenchen",
    "country_code": "DE"
  },
  "changes": [
    {"field": "name", "from": "  Example GmbH ", "to": "Example GmbH"},
    {"field": "road", "from": "Musterstr.", "to": "Musterstraße"},
    {"field": "country_code", "from": "de", "to": "DE"}
  ],
  "needs_review": []
}
```

When the input does not contain enough evidence to repair a field, the system
keeps the available value and names the field in `needs_review`. It does not
geocode, match entities, or create missing address values.

## Project pipeline

```text
dirty address record
        │
        ├── deterministic rules floor
        │
        ├── base MiniCPM5-1B
        │
        └── LoRA fine-tuned MiniCPM5-1B
                │
                ▼
validated JSON + field changes + review route
                │
                ▼
append-only audit record and paired-record evaluation
```

The three systems use the same output contract and are compared on the same
held-out records:

1. **Rules floor.** Deterministic normalization for well-understood cases.
2. **Base model.** Untuned MiniCPM5-1B with the production prompt and review
   policy.
3. **SFT model.** MiniCPM5-1B adapted with LoRA supervised fine-tuning.

## Output validation and audit

Responses must satisfy the fixed JSON Schema in
[`src/addr_repair/schema.py`](src/addr_repair/schema.py). Semantic validation
also checks that:

- every change agrees with the dirty input and returned clean record;
- a field is not changed and reviewed at the same time;
- duplicate changes and review fields are rejected;
- every changed field has an audit entry.

The audit JSONL records the input fingerprint, model and prompt revisions,
output, schema result, semantic result, latency, and score when paired gold
data is available.

## Results

The definitive v3 frozen test ran on 2026-09-15: the same 2,000
clean-gold-grouped records through all three systems, scored with the same
field-level scorer. The test manifest contains original dirty/clean pairs,
with no clean record shared across train, validation, and test.

| System | Precision | Recall | F1 | Damage | Empty-field fills | Unsupported additions | Review precision | Schema | Contract | p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| Rules floor | 0.9369 | 0.1033 | 0.1860 | 0.0 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0 ms |
| Base MiniCPM5 | 0.0281 | 0.0146 | 0.0192 | 0.0235 | 1,039 | 1 | 0.5445 | 0.7175 | 0.0 | 2,237.1 ms |
| v3 SFT MiniCPM5 | 0.6984 | 0.1991 | 0.3099 | 0.0237 | 1 | 88 | 0.9992 | 0.999 | 0.9535 | 1,443.7 ms |

The v3 SFT model improves repair F1 over the rules floor (0.3099 vs
0.1860) and the base model (0.0192), with much higher repair precision
than the base (0.6984 vs 0.0281). It makes one empty-field fill, compared
with 1,039 for the base. It still makes 88 unsupported additions, where a
prediction extends a non-empty input value with extra tokens. Damage is
0.0237, slightly above the base (0.0235) and above the rules floor (0.0).
The strongest v3 fields are road (precision 0.9233, recall 0.6066) and
house number (0.9322, 0.1378); name damage is the main weakness
(0.0754, 94 damaged clean fields).

One scoring rule shapes how to read recall. When a dirty field is
empty, the correct move is to leave it empty and flag it for review,
and the scorer still counts that as a miss. So recall 0.1991 means fixed
without guessing, not gaps closed. Review precision 0.9992 supports
that reading: when the model asks for a human, it is almost always
right to ask. Review record coverage is 0.809 for v3 SFT, compared with
0.810 for the rules floor and 0.754 for the base.

Two v3 records show both sides. The model over-edited the clean name
`Helmholtz-Gymnasium` to `Helmholtz-Gymnasium Karlsruhe`, copying the
locality into the name. On another record it held back correctly:
postcode empty in the input, left empty in the output and flagged in
`needs_review`, while `Wackerbarthstr.` was repaired to
`Wackerbarthstraße`.

Per-record audit logs stay local (gitignored). The committed result
files are `evals/frozen-test/{rules,base,sft}/metrics.json`; the shared
`freeze.json` reflects the last run, so compare the per-system files.
Training provenance is in `evals/sft-v3/`; the v3 GGUF is
`models/sft-Q4_K_M-clean-gold-v3.gguf`.
The metric definitions live in [`docs/EVAL.md`](docs/EVAL.md).

## Data and reproducibility

The address pairs come from the published
[Clean Me If You Can](https://github.com/D2IP-TUB/Clean-Me-If-You-Can) dataset.
The source data is derived from OpenStreetMap and is distributed under ODbL
1.0. Raw records are not committed to this repository. The definitive
`deduplicated_clean_gold_v1` split contains 3,540 / 728 / 1,424 distinct
clean records in the 5,000 / 1,000 / 2,000 train / validation / test rows,
pinned by manifest hashes. The test set contains only original source pairs.
Dataset manifests, checksums, fixtures, configuration revisions, and result
metadata provide the reproduction boundary.

Training follows the MiniCPM5 LoRA recipe through the v3 Colab runner:
TRL LoRA SFT on evidence-preserving targets, assistant-only loss, and
checkpoint selection on validation honesty metrics. The definitive run
used a T4 in fp16, selected `checkpoint-800` with score `0.7708`, and
trained for 4,574.6 seconds. Its adapter hash is
`7103451b9cc916920c59e5d68e8c3ada852b294be5eab55a0d1a9abdc77ec712`.
The v3 GGUF hash is
`889893112a0d7149c5df5ad0de0d928ccfd3d9d56011f2bed1e3f249d13ee54f`.
Merge and GGUF commands are in [`docs/REPRO.md`](docs/REPRO.md). Local serving uses
[`llama.cpp`](https://github.com/ggml-org/llama.cpp); base and tuned models
share prompt v2, decoder settings (temp 0.0, top_p 1.0, seed 7), context
2048, and Q4_K_M quantization.

## Repository layout

```text
configs/                  model, data, LoRA, and training settings
src/addr_repair/          rules, prompts, schema, audit, and scorer
scripts/                  data, training, evaluation, and report entry points
notebooks/                thin GPU training runner
fixtures/                 small checked-in smoke inputs
docs/                     split, evaluation, and reproduction notes
paper/                    technical report outline
tests/                    unit tests for rules, validation, and scoring
```

## Related documentation

- [`docs/EVAL.md`](docs/EVAL.md): metric and validation definitions
- [`docs/SPLITS.md`](docs/SPLITS.md): entity-disjoint data design
- [`docs/REPRO.md`](docs/REPRO.md): reproduction workflow and pinned inputs
- [`paper/outline.md`](paper/outline.md): technical report structure

## License

Code: Apache-2.0. Source address data: ODbL 1.0. See
[`LICENSE`](LICENSE) and the source dataset terms.
