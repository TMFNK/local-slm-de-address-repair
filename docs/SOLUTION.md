# Address repair as a business service (MbitAI)

Who this is for, what it delivers, and how an engagement works. The
technical detail lives in [`../paper/address-repair-techreport.pdf`](../paper/address-repair-techreport.pdf),
[`REPRO.md`](REPRO.md), and [`EVAL.md`](EVAL.md).

## The problem in one paragraph

Address data is often the quiet failure point in migrations, CRM imports,
master-data cleanup, and operational workflows. A normalizer that changes a
correct name or invents a missing postcode creates a harder problem than the
messy input. Cloud-model cleanup can improve coverage, but it may require
sending customer or partner data to a third party. MbitAI address repair is
designed around the other option: keep the data local, make every change
inspectable, and route unsupported cases to a human.

## What the service delivers

- **Validated address JSON.** Dirty German named-address records become a
  fixed, documented output contract rather than free-form model text.
- **A receipt for every decision.** Changed fields, input fingerprints,
  model revision, validation results, review fields, latency, and scores are
  recorded in an append-only audit trail.
- **A deterministic floor.** Rules handle the cases that can be justified
  mechanically; the local model is measured against that floor instead of
  replacing it blindly.
- **A review queue.** Missing evidence and uncertain repairs stay visible in
  `needs_review`. Unsupported additions and clean-field damage are scored
  separately.
- **A reproducible handover.** Pinned manifests, prompts, model hashes,
  fixtures, runbooks, and a frozen scorecard let a customer re-run the
  process after the engagement.

## Who buys it

- **Data and migration leads** who need cleaner imports without silently
  changing trusted fields.
- **CRM and master-data teams** that need field-level evidence for repairs,
  exceptions, and manual review.
- **Compliance and security teams** that do not want address records sent to
  a hosted model without a separate data-processing arrangement.
- **Consultancies and MSPs** that need a repeatable, customer-specific
  cleanup and handover pattern.

## Engagement shapes

1. **Assessment.** A representative sample is processed on the customer’s
   machine or an agreed private environment. MbitAI returns a repair
   scorecard, damage and unsupported-addition examples, review workload, and
   a go/no-go recommendation.
2. **Pilot.** The pipeline runs beside the current normalizer on a larger
   sample. We compare repair quality, harmful edits, review coverage, schema
   validity, and runtime on the customer’s records.
3. **Production handover.** MbitAI pins the configuration for the customer’s
   formats, creates golden fixtures and CI gates, documents the review route,
   and trains the team to operate and audit the pipeline.

The implementation can run locally through `llama.cpp`, or the deterministic
rules floor can be used on its own where that is the safer fit. The benchmark
uses public address records and does not make a production-safety claim; each
customer deployment needs its own acceptance set and review policy.

Contact: [MbitAI](https://www.mbitai.com)

## Measured v3 reference point

On 2,000 clean-gold-grouped public test records, the v3 SFT model reached
repair precision `0.6984`, recall `0.1991`, and F1 `0.3099`. It made one
empty-field fill and 88 unsupported additions, with damage rate `0.0237`,
schema validity `0.999`, contract validity `0.9535`, and review precision
`0.9992`. These figures describe the reference benchmark, not a promise
about a customer’s data.

## License and data boundary

The code is Apache-2.0. The source address data is ODbL-1.0 and is not
redistributed in this repository. Customer data stays under the customer’s
control; raw audit outputs must remain private when they contain sensitive
values.
