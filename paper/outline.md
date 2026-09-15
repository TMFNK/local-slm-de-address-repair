# Paper outline

1. Problem: address data breaks migration and ops work; silent changes beat visible review items for risk.
2. Method: rules floor, paired clean targets, MiniCPM5 LoRA SFT, schema check, audit log, review route.
3. Setup: ODbL source, entity-disjoint splits, pinned revisions, GPU training,
   and a shared local llama.cpp evaluation path.
4. Results: rules vs base vs v3 SFT on every metric in `docs/EVAL.md`.
   On the clean-gold v3 test, rules reaches F1 0.1860, base 0.0192, and
   v3 SFT 0.3099; v3 SFT repair precision is 0.6984 and review precision
   is 0.9992.
5. Failure exhibits: the v3 name over-edit
   (`Helmholtz-Gymnasium` → `Helmholtz-Gymnasium Karlsruhe`) and the
   correct review refusal for an empty postcode after repairing
   `Wackerbarthstr.` → `Wackerbarthstraße`.
6. Limits: public address data only, benchmark scope, no invented values, no production safety claim.
7. Repro: tag, lockfile, hashes, commands, raw artifacts. The selected v3
   checkpoint is `checkpoint-800`, with adapter SHA-256
   `7103451b9cc916920c59e5d68e8c3ada852b294be5eab55a0d1a9abdc77ec712`.
