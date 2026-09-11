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

Historical frozen test on 2026-09-10: the same 2,000 records through all
three systems, scored with the same field-level scorer. These results use the
pre-fix UUID-only split and must not be described as clean held-out
generalization. The tuned model
is LoRA checkpoint-939 (3 epochs, Colab T4, fp16), merged and exported
as Q4_K_M GGUF with pinned llama.cpp `b31b71f`.

| System | Precision | Recall | F1 | Damage | Inventions | Review precision | Schema | Contract | p50 |
|---|---|---|---|---|---|---|---|---|---|
| Rules floor | 0.9643 | 0.0744 | 0.1382 | 0.0 | 0 | 1.0 | 1.0 | 1.0 | 0 ms |
| Base MiniCPM5 | 0.0311 | 0.0092 | 0.0142 | 0.0169 | 898 | 0.5927 | 0.664 | 0.0 | 1620 ms |
| SFT MiniCPM5 | 0.8231 | 0.2438 | 0.3762 | 0.0114 | 0 | 0.9996 | 0.994 | 0.9665 | 1633 ms |

The tuned model repairs about three times what the rules floor repairs
(recall 0.24 vs 0.07, F1 0.38 vs 0.14) at lower precision (0.82 vs
0.96). It never fills in an empty field: 0 inventions against 898 for
the untuned base. Damage is 0.0114, between the rules floor (0.0) and
the base (0.0169). Strongest fields are road (precision 0.96, recall
0.61) and locality (0.93, 0.55). Most of the damage sits in the name
field (rate 0.0442, 55 of 87 events).

One scoring rule shapes how to read recall. When a dirty field is
empty, the correct move is to leave it empty and flag it for review,
and the scorer still counts that as a miss. So recall 0.24 means fixed
without guessing, not gaps closed. Review precision 0.9996 supports
that reading: when the model asks for a human, it is almost always
right to ask.

Two records show both sides. The model over-edited the clean name
`Pfarrhaus` to `Pfarrhaus Kalkhorst`, copying the locality into the
name. On another record it held back correctly: postcode empty in the
input, left empty in the output and flagged in `needs_review`, while
`Hardtstr.` on the same record was repaired to `Hardtstraße`.

Per-record audit logs stay local (gitignored). The committed result
files are `evals/frozen-test/{rules,base,sft}/metrics.json`; the shared
`freeze.json` reflects the last run, so compare the per-system files.
The metric definitions live in [`docs/EVAL.md`](docs/EVAL.md).

## Data and reproducibility

The address pairs come from the published
[Clean Me If You Can](https://github.com/D2IP-TUB/Clean-Me-If-You-Can) dataset.
The source data is derived from OpenStreetMap and is distributed under ODbL
1.0. Raw records are not committed to this repository. Pair-grouped splits
keep identical canonical dirty-plus-gold pairs in one partition and target
5,000 train / 1,000 validation / 2,000 test rows, pinned by manifest hashes.
Dataset manifests, checksums, fixtures, configuration revisions, and result
metadata provide the reproduction boundary.

Training follows the MiniCPM5 LoRA recipe through the thin Colab runner in
[`notebooks/colab_sft.ipynb`](notebooks/colab_sft.ipynb): TRL LoRA SFT on
evidence-preserving targets, assistant-only loss, checkpoint picked on
validation honesty metrics (checkpoint-939, score 0.769). The winning
adapter (`6aeb7a16…c9c8ba`) was merged and converted with pinned
llama.cpp `b31b71f`. Local serving uses
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
