# Reproduction

## Release boundary

The immutable software snapshot is the `v0.1.0` GitHub release. Its source
archive intentionally excludes raw source data, GGUF model files, and
per-record evaluation audits. The release also contains
`local-slm-de-address-repair-v0.1.0-manifests.tar.gz`, a supplementary archive
of the ignored manifests referenced by the committed metrics and run records.
Extract it under the repository root before running the frozen commands, or
regenerate the manifests from the pinned source dataset below. The release
asset SHA-256 is
`5f2e6ced37ffd5c72a951b0c4edc0b4b962e708c08a59579c47b1a7e0ef20ad0`.

## Current implementation

The repository is built in evaluation-first steps. Step 1 is runnable:

```bash
uv sync
uv run pytest
uv run ruff check src scripts tests
```

Data preparation is implemented but requires the pinned source release in
`data/raw/dirty.csv` and `data/raw/clean.csv`. After recording the release
metadata and checksums in `configs/data.yaml`, run:

```bash
uv run python scripts/prepare_data.py --config configs/data.yaml
```

The command refuses incomplete pairs, duplicate IDs, insufficient records,
unfillable duplicate-content groups, or accidental replacement of generated
manifests. It writes one manifest per split using complete canonical
dirty-plus-gold groups. Pass `--force` only when deliberately regenerating
those files. A fresh clone already has `fixtures/smoke_100/`, so Colab and
other clean checkouts need `--force` once.

The paired 100-record smoke fixture comes from the training partition
(`smoke_split: train`). It is safe to use for the rules-floor smoke gate and
prompt development. Keep the test partition unread until the one-shot frozen
evaluation.

The rules-floor smoke runner is the first experiment gate:

```bash
uv run python scripts/run_baseline.py --config configs/model.yaml
```

With gold pairs it also writes scored metrics next to the audit output:

```bash
uv run python scripts/run_baseline.py --pairs fixtures/smoke_100/pairs.jsonl \
  --out evals/baseline-smoke/audit.jsonl
```

This writes `evals/baseline-smoke/metrics.json` (aggregate plus per-field
repair precision/recall, damage rate, and review precision) alongside the
ignored audit JSONL. `--pairs` and `--fixture` are mutually exclusive.

Frozen evaluation runs every system through one shared local path
(parse, validate, audit, score). Start the pinned server first; the
`--reasoning off` flag is what engages no-think mode:

```bash
./llama.cpp/build/bin/llama-server -m models/MiniCPM5-1B-Q4_K_M.gguf -c 2048 --reasoning off
uv run python scripts/evaluate_local.py --system rules
uv run python scripts/evaluate_local.py --system base --manifest fixtures/smoke_100/manifest.json --out-dir evals/baseline-smoke/
```

Each run checks the manifest hash before scoring and writes ignored
`audit.jsonl` plus committed `metrics.json` and `freeze.json` (hashes and
aggregate numbers only, no raw records). SFT v1 needs the local GGUF and the
adapter hash from `evals/sft-v1/selection.json`:

```bash
uv run python scripts/evaluate_local.py --system sft \
  --gguf models/sft-Q4_K_M.gguf \
  --model-rev 6aeb7a169c76bbc18d351be78bfbdeb99e7cde557acbd17b140d26092c9c8ba9
```

The corrected pair-grouped model uses the v2 artifact and its own adapter
hash from `evals/sft-v2/selection.json`:

```bash
uv run python scripts/evaluate_local.py --system sft \
  --gguf models/sft-Q4_K_M-pair-grouped-v2.gguf \
  --model-rev beacfbfdcfc6cdbeca1576e8c07c0e6c1684dd72eb9fb968b888cc929a99d25b
```

The definitive clean-gold v3 run uses Drive root
`/content/drive/MyDrive/local-slm-de-address-repair-clean-gold-v3/`.
Its downloaded provenance is in `evals/sft-v3/`, the v3 manifests are in
`data/manifests/` (or the v0.1.0 supplementary manifest archive), and the
versioned GGUF is:

```bash
uv run python scripts/evaluate_local.py --system rules \
  --manifest data/manifests/test.json --out-dir evals/frozen-test-v3

uv run python scripts/evaluate_local.py --system sft \
  --gguf models/sft-Q4_K_M-clean-gold-v3.gguf \
  --model-rev 7103451b9cc916920c59e5d68e8c3ada852b294be5eab55a0d1a9abdc77ec712 \
  --manifest data/manifests/test.json --out-dir evals/frozen-test-v3
```

The base run uses the same command with `--system base` and the pinned base
GGUF. All three v3 runs use test-manifest records hash
`160eec15661de36d45265e03a053244311187d077b7c00e4d571d2294250350c`.
The v3 SFT GGUF SHA-256 is
`889893112a0d7149c5df5ad0de0d928ccfd3d9d56011f2bed1e3f249d13ee54f`.
The selected adapter is `checkpoint-800`, selected on validation only with
score `0.7708`; its full adapter SHA-256 is
`7103451b9cc916920c59e5d68e8c3ada852b294be5eab55a0d1a9abdc77ec712`.
The dedicated rerun result boundary is
`evals/frozen-test-v3/{rules,base,sft}/`; each directory contains
`metrics.json` and `runtime.json`, and the committed exhibit artifact is
`docs/exhibits/2026-09-15-v3-frozen-exhibits.md`.

The GGUF is gitignored. Put `models/sft-Q4_K_M.gguf` next to the base GGUF
(688,065,792 bytes, SHA-256
`c03e5b9c66fd43a8191ecf0eaa74351bc65e246ccef877db08d0944103703624`). The
2026-09-10 frozen-eval freeze file stored a 63-character prefix of the
adapter hash (`…c9c8ba`); the full SHA-256 ends in `…c9c8ba9`.

## Historical T4 SFT run (2026-09-10)

This is the run behind checkpoint-939 and the committed frozen-test SFT
metrics. It trained at commit `911cb9fe14101ecfa0ece8a2d67020aa0960ec6c` on
UUID-only splits. It is not a pair-grouped held-out result. A new Colab run
uses the current split code.

| Item | Value |
|---|---|
| Resolved config | `configs/train_t4_run.yaml` (fp16, `val_samples` 100, `val_gen_max_tokens` 256) |
| Ampere default | `configs/train_colab.yaml` (bf16, 200 val records) |
| GPU | Tesla T4, capability 7.5, torch 2.14.0+cu130 |
| Winner | `checkpoint-939`, selection score 0.769 |
| Adapter SHA-256 | `6aeb7a169c76bbc18d351be78bfbdeb99e7cde557acbd17b140d26092c9c8ba9` |
| Records | `evals/sft-v1/selection.json`, `evals/sft-v1/run.json` |
| Notebook | `notebooks/colab_sft.ipynb` |

Training finished (`train_runtime` 4556 s, `train_loss` 0.04755, eval loss
0.05969 → 0.04186). Honesty scoring was interrupted; `--select-only` scored
checkpoint-600, 800, and 939 and wrote `selection.json`. There is no
`training_record.json`.

Data came from Drive copies of the German named slice, prepared with
`--skip-archive --force`:

- dirty `2ef8ab1424c8c357af1aeaecef12b376cb7f20616437035f2753d0b2f2ac86ef`
- clean `78c852e3a5680a7460732b67ab87d82484a718297190fa8852af73a80817a192`

The notebook on Colab still picks a committed config from the GPU: T4 and
older get `configs/train_t4_run.yaml`; Ampere and newer get
`configs/train_colab.yaml`. The 2026-09-10 session derived the T4 file at
runtime with `patch_config.py`; that result is what is committed now.

### Merge and GGUF export

Exact commands from the T4 session after selection. Use `-j2` for the
llama.cpp build: unbounded `-j` ran out of memory on Colab.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id, rev = "openbmb/MiniCPM5-1B", "87179e5c1f455ef22e6223592d2d61351b525bfc"
out = "/content/drive/MyDrive/local-slm-de-address-repair/merged-sft"
tok = AutoTokenizer.from_pretrained(base_id, revision=rev)
base = AutoModelForCausalLM.from_pretrained(
    base_id, revision=rev, torch_dtype="auto", device_map="cpu"
)
model = PeftModel.from_pretrained(
    base,
    "/content/drive/MyDrive/local-slm-de-address-repair/checkpoints/checkpoint-939",
)
merged = model.merge_and_unload()
merged.save_pretrained(out)
tok.save_pretrained(out)
```

```bash
git clone https://github.com/ggml-org/llama.cpp /content/llama.cpp
cd /content/llama.cpp
git checkout b31b71f3a076bfc4278daad442203a9c51c6e676
cmake -B build -DGGML_CUDA=OFF
cmake --build build --config Release -j2 --target llama-quantize

uv run --with gguf --with sentencepiece python /content/llama.cpp/convert_hf_to_gguf.py \
  /content/drive/MyDrive/local-slm-de-address-repair/merged-sft \
  --outfile /content/drive/MyDrive/local-slm-de-address-repair/sft-f16.gguf \
  --outtype f16

/content/llama.cpp/build/bin/llama-quantize \
  /content/drive/MyDrive/local-slm-de-address-repair/sft-f16.gguf \
  /content/drive/MyDrive/local-slm-de-address-repair/sft-Q4_K_M.gguf \
  Q4_K_M
```

Quantize reported 2061.29 MiB F16 → 651.29 MiB Q4_K_M. Download
`sft-Q4_K_M.gguf` from Drive to `models/` on the eval machine. Adapter
weights, merged checkpoints, and GGUF files stay out of git; hashes live in
`evals/sft-v1/` and the frozen-eval metric files.

## Corrected pair-grouped SFT run (2026-09-14, SUPERSEDED)

This rerun trained on the first pair-grouped manifests, whose largest-first
allocator filled train/val/test with 4/3/2 distinct pairs (see
`docs/SPLITS.md`). The model trained on 4 distinct addresses and cannot
generalize; its validation tie (all checkpoints abstain, repair F1 0.0)
already showed this. Do not evaluate or report this artifact. It stays in
`evals/sft-v2/` and `models/sft-Q4_K_M-pair-grouped-v2.gguf` for the record
only.

Provenance of the retired run (kept, not valid): commit
`f9b0ca6d0720c8dfd90a9ffd8907c729a68bc38e`, fresh run, winner
`checkpoint-200`, selection score 0.6667 (three-way tie with 800 and 939),
4339.0 s train plus 1933.3 s val scoring on a Tesla T4 (fp16), adapter
`beacfbfd…99d25b`, manifests train `a2f810d0…` / val `9b64d693…`, local
GGUF `models/sft-Q4_K_M-pair-grouped-v2.gguf` (688,065,792 bytes,
`3189aeac…`), Drive root
`/content/drive/MyDrive/local-slm-de-address-repair-pair-grouped-v2`.

## Intermediate seeded-shuffle split (2026-09-14, superseded by v3)

The intermediate manifests used the seeded-shuffle allocator (`split_seed` 7):

| Split | Rows | Distinct pairs | Manifest records SHA-256 |
|---|---|---|---|
| train | 5,000 | 956 | `bde07df721c9d8460417ab0191a7c244fb1649a0f2f797af0bbe3c63b134bc90` |
| val | 1,000 | 189 | `b87ba19524901b1db4696a1d95a3f727f4bed0d0b5f06972a33231b70d3daf16` |
| test | 2,000 | 367 | `a6705579e5c1a0cedc0cde34267d122d1eb475c53659073626719ad31b09c0b1` |

No fingerprint crosses splits; smoke 100 is in train and outside test; all
six defect types occur in every split. Rules floor on the intermediate test
manifest: precision 0.9927, recall 0.1827, F1 0.3087, damage 0.0. The
superseded generic `evals/frozen-test/` copy is not a separate result; the
definitive v3 result is in `evals/frozen-test-v3/`.

## Clean-gold v3 run (2026-09-15)

The v3 notebook used commit `3464395ac72289265750096551329c038b0c8061`,
a fresh Drive root, and a Tesla T4 in fp16. It selected `checkpoint-800`
with validation honesty score `0.7708`; training took 4,574.6 seconds and
validation scoring took 2,364.9 seconds. The definitive split uses
`deduplicated_clean_gold_v1`: 3,540 / 728 / 1,424 distinct clean records
in train / validation / test, with records hashes:

- train `f7a3defaab7aa898dbc320a398f58e317dbc6bfd7435c5c7a29f5f27f2a808cd`
- validation `e331cb5ac908827b00698b45aefa8f0f8f13df1d9e476f5ab7a384d21b8008bf`
- test `160eec15661de36d45265e03a053244311187d077b7c00e4d571d2294250350c`

The v3 adapter SHA-256 is
`7103451b9cc916920c59e5d68e8c3ada852b294be5eab55a0d1a9abdc77ec712`.
The local GGUF SHA-256 is
`889893112a0d7149c5df5ad0de0d928ccfd3d9d56011f2bed1e3f249d13ee54f`.
Provenance is preserved in `evals/sft-v3/`.

Frozen test results:

- rules: precision `0.9369`, recall `0.1033`, F1 `0.1860`, damage `0.0`,
  schema/semantic/contract validity `1.0/1.0/1.0`;
- base: precision `0.0281`, recall `0.0146`, F1 `0.0192`, damage `0.0235`,
  `1,039` empty-field fills, schema/semantic/contract validity
  `0.7175/0.0/0.0`;
- v3 SFT: precision `0.6984`, recall `0.1991`, F1 `0.3099`, damage
  `0.0237`, one empty-field fill and `88` unsupported additions,
  schema/semantic/contract validity `0.999/0.9535/0.9535`.

Runtime probes measured server launch to `/health`, then sampled resident
memory for one second after load: base model load `1,561.8 ms`, peak RSS
`810.6 MB`; v3 SFT model load `661.7 ms`, peak RSS `816.8 MB`. Rules have
no model server and used `31.4 MB` evaluator RSS. These are startup-memory
measurements; per-record latency remains the production-path runtime metric.

The two report exhibits are a name over-edit
(`Helmholtz-Gymnasium` → `Helmholtz-Gymnasium Karlsruhe`) and a correct
abstention: `Wackerbarthstr.` → `Wackerbarthstraße` while the empty postcode
is left empty and sent to review.

`make_report.py` generates a Markdown summary from committed metrics; it is
not itself a result boundary. The committed per-system metrics and freeze
metadata are the result boundary; older UUID-only and superseded pair-grouped
runs remain historical only.

## Targeted name/locality SFT v5p1 (2026-09-18, frozen eval complete)

One bounded epoch from the frozen clean-gold-v3 adapter, on the audited
targeted view only: every pinned-train pair where `name` or `locality`
differs (2,798 rows) plus every all-clean train pair as damage control
(274 rows). Validation and test stayed pinned.

| Item | Value |
|---|---|
| Commit | `e966c0cbd6819abe2a6ff993187b9f268f42b70a`, fresh run (`resumed: false`) |
| GPU | Tesla T4, capability 7.5, torch 2.14.0+cu130, fp16, seed 7 |
| Source adapter | clean-gold-v3 `checkpoint-800`, SHA-256 `7103451b…77ec712` |
| Targeted train | 3,072 rows, records hash `78c1a680e9a854ab4b142833faa16c699db047a3469a0b909aaa900ac17b22f9` |
| Val / test | `e331cb5ac908827b00698b45aefa8f0f8f13df1d9e476f5ab7a384d21b8008bf` / `160eec15661de36d45265e03a053244311187d077b7c00e4d571d2294250350c` |
| Winner | `checkpoint-150`, selection score `0.7626` (runner-up `checkpoint-192` at `0.7609`) |
| Winner adapter SHA-256 | `3494ac23b69554e724c83c72d949f7352da2584cb53c1ad28eb6b34cc6a09397` |
| Winner val metrics | review precision `1.0`, damage `0.0391`, repair F1 `0.3269` (P `0.6788` / R `0.2153`), schema `0.995`, contract `0.965` |
| Train / scoring time | 1,109.6 s train, 5,231.4 s val scoring |
| Records | `evals/sft-v5p1/selection.json`, `training_record.json`, `resolved_config.yaml` |
| Local GGUF | `models/sft-Q4_K_M-targeted-name-locality-v5p1.gguf` (688,065,824 bytes, SHA-256 `1d45c3d86d8e33c03967bcebb46d15a606abfef9e91538e75f3b0e7f1cd1fe08`) |
| Drive root | `/content/drive/MyDrive/local-slm-de-address-repair/checkpoints-sft-targeted-name-locality-v5p1` |
| Notebook | vault `docs/collab notebook/colab_sft_targeted_name_locality_v5p1.ipynb` |

Do not use the quarantined v1 Drive dir
(`checkpoints-sft-targeted-name-locality-v1`, with `checkpoint-300` /
`checkpoint-313`): that run trained on the full 5,000-row split after the
v5 notebook left `data_targeted_name_locality.yaml` pointing at
`data/manifests/`. It is not a targeted artifact.

The v5p1 evaluation command, using the same frozen path as v3, was:

```bash
uv run python scripts/evaluate_local.py --system sft \
  --gguf models/sft-Q4_K_M-targeted-name-locality-v5p1.gguf \
  --model-rev 3494ac23b69554e724c83c72d949f7352da2584cb53c1ad28eb6b34cc6a09397 \
  --manifest data/manifests/test.json --out-dir evals/frozen-test-v5p1
```

The model run wrote all 2,000 audit rows, but scoring initially stopped on
row 233 because the model returned change objects in `needs_review` instead
of field-name strings. The schema correctly rejected that row. The scorer
now skips non-string review entries without converting them into review
fields. A regression test covers the malformed shape, and the preserved
audit was rescored offline without contacting the server.

The offline result boundary is
`evals/frozen-test-v5p1/{sft/metrics.json,freeze.json}`. The audit contained
one malformed review row and four counted server-error rows, and passed all
manifest, record-order, dirty-payload, and input-fingerprint checks. The
audit still contains all 2,000 input rows; the four transient HTTP 500
responses are represented as unusable outputs in the committed metrics.
Results:

- repair precision `0.2619`, recall `0.1354`, F1 `0.1785`;
- damage `0.1688`, inventions `5`, unsupported additions `288`;
- schema / semantic / contract validity `0.796 / 0.393 / 0.393`;
- review precision `0.9743`, review field rate `0.1789`, record coverage
  `0.675`;
- median / p95 record latency `2,093.1 / 3,191.9 ms`.

This result is experimental and does not replace the shipped v3 result.

## Pin record

Record Python, TRL, Transformers, PEFT, `llama.cpp` revision, MiniCPM5
revision, dataset release, prompt revision, decoder settings, and checkpoint
hashes in configs, manifests, `evals/sft-v1/`, and `uv.lock`.
