# Paper outline

1. Problem: address data breaks migration and ops work; silent changes beat visible review items for risk.
2. Method: rules floor, paired clean targets, MiniCPM5 LoRA SFT, schema check, audit log, review route.
3. Setup: ODbL source, entity-disjoint splits, pinned revisions, GPU training,
   and a shared local llama.cpp evaluation path.
4. Results: rules vs base vs SFT on every metric in `docs/EVAL.md`.
5. Failure exhibits: one wrong repair + one correct review refusal, with audit records.
6. Limits: public address data only, benchmark scope, no invented values, no production safety claim.
7. Repro: tag, lockfile, hashes, commands, raw artifacts.
