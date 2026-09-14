# Splits

Pair-grouped splits by canonical dirty-plus-gold fingerprint, dealt in
seeded-shuffle order over whole groups:

- train: 5,000 entities (may add controlled corruptions from clean rows)
- val: 1,000 entities (checkpoint choice only)
- test: 2,000 entities, original paired dirty/clean rows only

Every ID sharing the same six-field dirty-plus-gold pair stays in one split.
The preparation command assigns complete groups only, with exact row counts,
and fails if the requested counts cannot be reached without splitting a
group. The group order is shuffled with `splits.split_seed` (default 7,
recorded in every manifest), so each split spans many different contents;
the same seed always gives the same splits. A diversity guard refuses splits
where any split holds fewer distinct pairs than 10% of its rows, with an
error naming the seed to change.

The source slice repeats boilerplate addresses thousands of times, so rows
are not independent: the independent units are distinct pairs, which never
cross split boundaries. Uncertainty analysis must count distinct pairs,
not rows.

Split history: the first pair-grouped code ordered groups largest-first.
On 2026-09-14 that filled train/val/test with 4/3/2 distinct pairs from the
duplicated head of the data, which made training and evaluation vacuous.
The seeded shuffle replaced it; seed 7 gives 956/189/367 distinct pairs
for train/val/test with all six defect types present in every split.
Manifests in `data/manifests/` record IDs, source-file hashes, source URL,
dataset date, split seed, and the canonical paired record hash.

The preparation command reads `data/raw/dirty.csv` and
`data/raw/clean.csv`, checks that rows are aligned by `id`, and projects both
files to the six fields in the output contract. `country_code` is
canonicalized to uppercase ISO alpha-2 (`de` becomes `DE`) in the same
projection step, matching the JSON Schema.

The 100-record smoke fixture comes from the training partition
(`smoke_split: train`). It is used for the rules-floor smoke gate and prompt
development only. The final test partition remains reserved for the one-shot
evaluation and is not read during prompt development.
