# Eval

Same 2,000 frozen test records for rules floor, base MiniCPM5, and SFT MiniCPM5.

- Repair precision / recall / F1 (field level, vs paired gold)
- Damage rate (clean fields changed wrongly / all clean fields)
- Review precision + coverage
- JSON-schema validity
- Cost: model-load time, median + p95 record time, peak memory on M2 Air

Exact-value and normalized-value scores stay separate when they disagree. After reading the frozen result: no prompt, normalizer, model, or test-data changes.
