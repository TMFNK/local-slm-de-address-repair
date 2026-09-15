# Splits

The definitive v3 split groups by the canonical clean six-field record,
deduplicates exact dirty-plus-gold rows, and deals groups in seeded-shuffle
order:

- train: 5,000 entities (may add controlled corruptions from clean rows)
- val: 1,000 entities (checkpoint choice only)
- test: 2,000 entities, original paired dirty/clean rows only

Every ID sharing the same clean record stays in one split, and exact
dirty-plus-gold duplicates are reduced to one representative before
allocation. The v3 manifests use `deduplicated_clean_gold_v1` with
`split_seed` 7 and contain 3,540 / 728 / 1,424 distinct clean records in
train / validation / test. The test partition uses only original source
pairs. The manifest records hashes are `f7a3defa…`, `e331cb5a…`, and
`160eec15…` respectively.

The source slice repeats boilerplate addresses thousands of times, so rows
are not independent: the independent units are distinct pairs, which never
cross split boundaries. Uncertainty analysis must count distinct pairs,
not rows.

Split history: the first pair-grouped code ordered groups largest-first.
On 2026-09-14 that filled train/val/test with 4/3/2 distinct pairs from the
duplicated head of the data, which made training and evaluation vacuous.
The seeded shuffle was an intermediate fix; the clean-gold grouping replaced
it for the definitive v3 benchmark. Manifests in `data/manifests/` record
IDs, source-file hashes, source URL, dataset date, split seed, grouping mode,
and the canonical paired record hash.

The preparation command reads `data/raw/dirty.csv` and
`data/raw/clean.csv`, checks that rows are aligned by `id`, and projects both
files to the six fields in the output contract. `country_code` is
canonicalized to uppercase ISO alpha-2 (`de` becomes `DE`) in the same
projection step, matching the JSON Schema.

The 100-record smoke fixture comes from the training partition
(`smoke_split: train`). It is used for the rules-floor smoke gate and prompt
development only. The final test partition remains reserved for the one-shot
evaluation and is not read during prompt development.
