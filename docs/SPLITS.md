# Splits

Pair-grouped splits by canonical dirty-plus-gold fingerprint, ordered
deterministically by fingerprint and group size:

- train: 5,000 entities (may add controlled corruptions from clean rows)
- val: 1,000 entities (checkpoint choice only)
- test: 2,000 entities, original paired dirty/clean rows only

Every ID sharing the same six-field dirty-plus-gold pair stays in one split.
The preparation command assigns complete groups only and fails if the
requested row counts cannot be reached without splitting a group. Manifests in
`data/manifests/` record IDs, source-file hashes, source URL, dataset date,
and the canonical paired record hash.

The preparation command reads `data/raw/dirty.csv` and
`data/raw/clean.csv`, checks that rows are aligned by `id`, and projects both
files to the six fields in the output contract. `country_code` is
canonicalized to uppercase ISO alpha-2 (`de` becomes `DE`) in the same
projection step, matching the JSON Schema.

The smoke fixture still comes from the test partition in this intermediate
revision. It must not be used for prompt development; the next fix moves it
to training or validation before the new experiment starts.
