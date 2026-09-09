# Splits

Entity-disjoint splits by address id, sorted for determinism:

- train: 5,000 entities (may add controlled corruptions from clean rows)
- val: 1,000 entities (checkpoint choice only)
- test: 2,000 entities, original paired dirty/clean rows only

No entity or clean target crosses splits. Manifests in `data/manifests/` record
IDs, source-file hashes, source URL, dataset date, and the canonical paired
record hash. The preparation command reads `data/raw/dirty.csv` and
`data/raw/clean.csv`, checks that rows are aligned by `id`, and projects both
files to the six fields in the output contract. The smoke fixture
(`fixtures/smoke_100/`) holds 100 original dirty/clean pairs from the test
partition; check 20 by hand before training.
