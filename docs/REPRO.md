# Repro

1. `uv sync`
2. Download the Zenodo release from `configs/data.yaml`; record URL, date, checksum.
3. `uv run python scripts/prepare_data.py --config configs/data.yaml`
4. `uv run python scripts/run_baseline.py --config configs/model.yaml` (smoke gate)
5. Open `notebooks/colab_sft.ipynb` on Colab GPU; it calls `scripts/train_sft.py`. Save checkpoint hashes.
6. Convert + quantize adapter to GGUF Q4 with the commands in the notebook.
7. `uv run python scripts/evaluate_local.py --config configs/model.yaml` on the M2 Air.
8. `uv run python scripts/make_report.py --evals evals/frozen-test/`

Pin record: Python, TRL, Transformers, PEFT, llama.cpp rev, MiniCPM5 rev, dataset release — all in configs + manifests + `uv.lock`.
