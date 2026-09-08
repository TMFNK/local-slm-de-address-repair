# Splits

Entity-disjoint splits by address id, sorted for determinism:

- train: 5,000 entities (may add controlled corruptions from clean rows)
- val: 1,000 entities (checkpoint choice only)
- test: 2,000 entities, original paired dirty/clean rows only

No entity or clean target crosses splits. Manifests in `data/manifests/` record ids, hashes, source URL, archive checksum, and date. The smoke fixture (`fixtures/smoke_100/`) holds 100 records; check 20 by hand before training.
