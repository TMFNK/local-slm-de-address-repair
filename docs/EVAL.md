# Eval

Same 2,000 frozen test records for rules floor, base MiniCPM5, and SFT MiniCPM5.

- Repair precision / recall / F1 (field level, vs paired gold)
- Damage rate (clean fields changed wrongly / all clean fields)
- Review precision
- `review_field_rate` (flagged fields / all field slots)
- `review_record_coverage` (records with at least one review flag / all records)
- JSON-schema validity
- Semantic validity (no semantic errors), parse rate, contract validity
  (schema AND semantics — the usable-output rate)
- Empty-field fills / empty-field fill rate: fields filled from an empty dirty
  input. The historical metric keys are `inventions` and `invention_rate`.
  A correct repair requires a non-empty dirty field — filling an empty field
  never counts as a true positive, even when the guess matches gold.
- `unsupported_additions` / `unsupported_addition_rate`: predictions that
  retain a non-empty dirty value as a contiguous token sequence while adding
  extra tokens. This diagnostic catches unsupported additions such as
  `Pfarrhaus` → `Pfarrhaus Kalkhorst`.
  Correct abstention (kept empty + flagged) still counts as a recall miss.
- Runtime: model-load time, median and p95 record time, and peak memory

Exact-value and normalized-value scores stay separate when they disagree. After reading the frozen result: no prompt, normalizer, model, or test-data changes.

The scorer also reports per-field counts for correct repairs, wrong repairs,
missed repairs, correct abstentions, and clean-field damage. Review precision
means that the reviewed input field had no usable value in the dirty record;
it is not inferred from whether the paired gold record happens to be empty.

JSON Schema validation checks response shape and types. Semantic validation
then checks that each change agrees with the dirty input and returned
`clean_record`, that reviewed fields are unchanged, and that every changed
field has an audit entry.
