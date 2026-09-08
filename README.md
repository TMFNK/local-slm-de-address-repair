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

## Evaluation

The paired source data provides dirty and clean records. The evaluation
measures:

- repair precision, recall, and F1;
- clean-field damage rate;
- correct abstentions and review precision;
- schema and semantic validity;
- per-field error counts;
- model-load time, record latency, and peak memory.

Exact-value and documented normalized-value scores are kept separate when they
disagree. Results and failure exhibits will be added here after the frozen
evaluation runs.

The metric definitions live in [`docs/EVAL.md`](docs/EVAL.md).

## Data and reproducibility

The address pairs come from the published
[Clean Me If You Can](https://github.com/D2IP-TUB/Clean-Me-If-You-Can) dataset.
The source data is derived from OpenStreetMap and is distributed under ODbL
1.0. Raw records are not committed to this repository. Dataset manifests,
checksums, fixtures, configuration revisions, and result metadata provide the
reproduction boundary.

Training uses the MiniCPM5 LoRA recipe and the thin Colab runner in
[`notebooks/colab_sft.ipynb`](notebooks/colab_sft.ipynb). Local serving uses
[`llama.cpp`](https://github.com/ggml-org/llama.cpp). Base and adapted models
must use the same prompt, decoder settings, runtime revision, and conversion
procedure.

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
