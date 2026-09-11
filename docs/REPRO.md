# Reproduction

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
aggregate numbers only, no raw records). SFT needs the local GGUF and the
adapter hash from `evals/sft-v1/selection.json`:

```bash
uv run python scripts/evaluate_local.py --system sft \
  --gguf models/sft-Q4_K_M.gguf \
  --model-rev 6aeb7a169c76bbc18d351be78bfbdeb99e7cde557acbd17b140d26092c9c8ba9
```

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

## Still open

`make_report.py` is still a placeholder. Do not treat a printed placeholder
as an experiment result. Repeat training, selection, export, and frozen
evaluation on the pair-grouped split before calling the scores clean
held-out generalization.

## Pin record

Record Python, TRL, Transformers, PEFT, `llama.cpp` revision, MiniCPM5
revision, dataset release, prompt revision, decoder settings, and checkpoint
hashes in configs, manifests, `evals/sft-v1/`, and `uv.lock`.
