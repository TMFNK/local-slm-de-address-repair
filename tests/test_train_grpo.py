import csv
import json
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import train_grpo

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def _row(road="Musterstraße", postcode="80331"):
    # Six contract fields only: load_paired_records strips rows to FIELDS,
    # so fixtures must too (an "id" inside the record breaks the schema).
    return {
        "name": "Example GmbH",
        "road": road,
        "house_number": "12",
        "postcode": postcode,
        "locality": "München",
        "country_code": "DE",
    }


def _csv_row(entity_id, **overrides):
    return {"id": entity_id, **_row(**overrides)}


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def _setup(tmp_path):
    """Write raw CSVs, manifests, and configs; return GRPO train config path."""
    from addr_repair.io import load_paired_records, records_hash, sha256_file

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    dirties = [
        _csv_row("e1", road="Musterstr."),  # hard: needs repair
        _csv_row("e2", postcode=""),  # hard: needs abstention path (gold filled)
        _csv_row("e3"),  # no-op: dirty already equals gold
        _csv_row("e4", road="musterstrasse"),  # hard: needs repair
    ]
    golds = [_csv_row("e1"), _csv_row("e2"), _csv_row("e3"), _csv_row("e4")]
    _write_csv(raw_dir / "dirty.csv", dirties)
    _write_csv(raw_dir / "clean.csv", golds)
    records = load_paired_records(raw_dir / "dirty.csv", raw_dir / "clean.csv")
    by_id = {r["id"]: r for r in records}
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()

    def _manifest(name, ids):
        recs = [by_id[i] for i in ids]
        (manifests_dir / f"{name}.json").write_text(
            json.dumps(
                {
                    "entity_ids": ids,
                    "records_sha256": records_hash(recs),
                    "dirty_sha256": sha256_file(raw_dir / "dirty.csv"),
                    "clean_sha256": sha256_file(raw_dir / "clean.csv"),
                }
            ),
            encoding="utf-8",
        )

    _manifest("train", ["e1", "e2", "e3", "e4"])
    _manifest("val", ["e3"])
    _manifest("test", [])
    (tmp_path / "model.yaml").write_text(
        "model_id: fake/MiniCPM5-1B\nmodel_rev: deadbeef\nprompt_rev: v2\n"
        "chat_template_mode: no-think\ndecode: {temperature: 0.0, top_p: 1.0, max_tokens: 512, seed: 7}\n",
        encoding="utf-8",
    )
    (tmp_path / "lora.yaml").write_text(
        "r: 16\nlora_alpha: 32\nlora_dropout: 0.05\n"
        "target_modules: [q_proj, v_proj]\nassistant_only_loss: true\nseed: 7\n",
        encoding="utf-8",
    )
    (tmp_path / "data.yaml").write_text(
        f"dirty_file: dirty.csv\nclean_file: clean.csv\nid_field: id\n"
        f"raw_dir: {raw_dir}\nmanifests_dir: {manifests_dir}\n",
        encoding="utf-8",
    )
    train_cfg = tmp_path / "train_grpo.yaml"
    train_cfg.write_text(
        f"base_config: {tmp_path / 'model.yaml'}\nlora_config: {tmp_path / 'lora.yaml'}\n"
        f"data_config: {tmp_path / 'data.yaml'}\n"
        f"init_adapter_path: {tmp_path / 'adapter'}\noutput_dir: {tmp_path / 'checkpoints'}\n"
        "per_device_train_batch_size: 2\ngradient_accumulation_steps: 2\nlearning_rate: 0.000001\n"
        "max_steps: 10\nbf16: false\nfp16: false\ntemperature: 0.7\nnum_generations: 4\n"
        "max_completion_length: 64\nbeta: 0.01\ngradient_checkpointing: true\n"
        "logging_steps: 5\nsave_steps: 10\nsave_total_limit: 1\nreport_to: none\nseed: 7\n"
        "trust_remote_code: true\ntrain_rows: 2\ntrain_seed: 7\n"
        "val_samples: 1\nval_gen_max_tokens: 32\n",
        encoding="utf-8",
    )
    return train_cfg


def test_select_grpo_rows_excludes_noops_and_is_deterministic(tmp_path):
    import train_sft

    cfg = _setup(tmp_path)
    resolved = train_sft.load_resolved_config(cfg)
    pairs, _ = train_sft.load_split_pairs(resolved, "train")
    rows_a, stats = train_grpo.select_grpo_rows(pairs, 2, seed=7)
    rows_b, _ = train_grpo.select_grpo_rows(pairs, 2, seed=7)
    assert stats["train_pairs"] == 4
    assert stats["hard_pairs"] == 3  # e3 (no-op) excluded
    assert stats["picked"] == 2
    assert [r["id"] for r in rows_a] == [r["id"] for r in rows_b]
    assert all(set(r) == {"id", "prompt", "dirty", "gold"} for r in rows_a)
    assert rows_a[0]["prompt"][0]["role"] == "system"


def test_grpo_reward_scores_good_above_broken():
    dirty = _row(road="Musterstr.")
    gold = _row()
    good = json.dumps(
        {
            "clean_record": gold,
            "changes": [{"field": "road", "from": "Musterstr.", "to": "Musterstraße"}],
            "needs_review": [],
        }
    )
    scores = train_grpo.grpo_reward(["p", "p"], [good, "not json"], [dirty, dirty], [gold, gold])
    assert scores == [0.22, 0.0]


def test_completion_unwrap_handles_trl_message_lists():
    as_text = train_grpo._completion_to_text
    assert as_text("raw") == "raw"
    assert as_text([[{"role": "assistant", "content": "hello"}]]) == "hello"
    assert as_text([{"role": "assistant", "content": "hi"}]) == "hi"
    assert as_text({"role": "assistant", "content": "yo"}) == "yo"
    assert as_text([]) == ""


def test_grpo_reward_accepts_trl_message_list_completions():
    dirty = _row(road="Musterstr.")
    gold = _row()
    good = json.dumps(
        {
            "clean_record": gold,
            "changes": [{"field": "road", "from": "Musterstr.", "to": "Musterstraße"}],
            "needs_review": [],
        }
    )
    wrapped = [[{"role": "assistant", "content": good}]]
    scores = train_grpo.grpo_reward(["p"], [wrapped], [dirty], [gold])
    assert scores == [0.22]


def test_build_grpo_config_constructs_real_config(tmp_path):
    import train_sft

    cfg = _setup(tmp_path)
    resolved = train_sft.load_resolved_config(cfg)
    args, dropped = train_grpo.build_grpo_config(resolved["train"], tmp_path / "out")
    assert dropped == []
    assert args.num_generations == 4
    assert args.max_completion_length == 64
    assert args.max_steps == 10
    assert args.beta == 0.01


def test_build_grpo_config_reports_unsupported_kwargs(tmp_path, monkeypatch):
    import train_sft
    from trl import GRPOConfig

    cfg = _setup(tmp_path)
    resolved = train_sft.load_resolved_config(cfg)
    fields = dict(GRPOConfig.__dataclass_fields__)
    removed = fields.pop("beta")
    monkeypatch.setattr(GRPOConfig, "__dataclass_fields__", fields)
    try:
        _, dropped = train_grpo.build_grpo_config(resolved["train"], tmp_path / "out")
    finally:
        fields["beta"] = removed
    assert dropped == ["beta"]


def test_dry_run_ok(tmp_path, monkeypatch, capsys):
    cfg = _setup(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_grpo.py", "--config", str(cfg), "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        train_grpo.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "picked=2" in out
    assert "group=4" in out
    assert "test overlap: none" in out
    assert "mean(review_precision, 1-damage, F1)" in out


def test_missing_adapter_refuses_training(tmp_path):
    import train_sft

    cfg = _setup(tmp_path)
    resolved = train_sft.load_resolved_config(cfg)
    with pytest.raises(SystemExit, match="adapter not found"):
        train_grpo.run_training(resolved)


def test_adapter_parity_accepts_v3_shape(tmp_path, capsys):
    from types import SimpleNamespace

    import train_sft

    cfg = _setup(tmp_path)
    resolved = train_sft.load_resolved_config(cfg)
    active = SimpleNamespace(r=16, lora_alpha=32, target_modules=["v_proj", "q_proj"])
    train_grpo.assert_adapter_parity(active, resolved["lora"], tmp_path / "adapter")
    assert "continuing v3 adapter in place" in capsys.readouterr().out


def test_adapter_parity_refuses_foreign_shape(tmp_path):
    from types import SimpleNamespace

    import train_sft

    cfg = _setup(tmp_path)
    resolved = train_sft.load_resolved_config(cfg)
    foreign = SimpleNamespace(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    with pytest.raises(SystemExit, match="foreign adapter"):
        train_grpo.assert_adapter_parity(foreign, resolved["lora"], tmp_path / "adapter")


def test_run_training_continues_adapter_trainable_without_fresh_config(tmp_path, monkeypatch):
    """Regression: PeftModel defaults to frozen; TRL refuses PeftModel+peft_config.

    So run_training must load with is_trainable=True and must not hand a
    fresh peft_config to GRPOTrainer. Heavy GPU pieces are stubbed; the wiring
    (kwargs on both calls) is asserted for real.
    """
    from types import SimpleNamespace

    import peft
    import train_sft
    import transformers
    import trl

    cfg = _setup(tmp_path)
    (tmp_path / "adapter").mkdir()
    resolved = train_sft.load_resolved_config(cfg)
    calls = {}

    class FakeTokenizer:
        chat_template = None
        pad_token = None
        eos_token = "</s>"

    class FakeBaseModel:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)

        def gradient_checkpointing_enable(self, **kwargs):
            pass

    class FakeAdapterModel(FakeBaseModel):
        def __init__(self):
            super().__init__()
            self.peft_config = {
                "default": SimpleNamespace(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
            }

    @classmethod
    def _fake_peft_from_pretrained(cls, model, path, **kwargs):
        calls["peft_kwargs"] = kwargs
        return FakeAdapterModel()

    class FakeGRPOTrainer:
        def __init__(self, **kwargs):
            calls["trainer_kwargs"] = kwargs

        def train(self, resume_from_checkpoint=None):
            calls["resume"] = resume_from_checkpoint

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: FakeTokenizer()),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: FakeBaseModel()),
    )
    monkeypatch.setattr(transformers, "set_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", _fake_peft_from_pretrained)
    monkeypatch.setattr(trl, "GRPOTrainer", FakeGRPOTrainer)
    monkeypatch.setattr(
        train_sft, "_iter_candidates", lambda out: [tmp_path / "checkpoints" / "checkpoint-1"]
    )
    metrics = {
        "review_precision": 1.0,
        "damage_rate": 0.0,
        "repair_f1": 0.2,
        "contract_validity": 0.9,
        "inventions": 0,
    }
    monkeypatch.setattr(train_sft, "score_candidate", lambda *args, **kwargs: dict(metrics))

    record = train_grpo.run_training(resolved)

    assert calls["peft_kwargs"].get("is_trainable") is True
    assert "peft_config" not in calls["trainer_kwargs"]
    assert record["continued_adapter_in_place"] is True
    assert record["winner"] == "checkpoint-1"
    assert record["mode"] == "grpo-v5"


def test_assert_no_test_overlap_allows_disjoint_ids(tmp_path):
    import train_sft

    cfg = _setup(tmp_path)
    resolved = train_sft.load_resolved_config(cfg)
    pairs, _ = train_sft.load_split_pairs(resolved, "train")
    rows, _ = train_grpo.select_grpo_rows(pairs, 2, seed=7)
    train_grpo.assert_no_test_overlap(rows, set())
    train_grpo.assert_no_test_overlap(rows, {"not-in-train"})


def test_assert_no_test_overlap_aborts_on_leaked_id(tmp_path):
    import train_sft

    cfg = _setup(tmp_path)
    resolved = train_sft.load_resolved_config(cfg)
    pairs, _ = train_sft.load_split_pairs(resolved, "train")
    rows, _ = train_grpo.select_grpo_rows(pairs, 2, seed=7)
    leaked = rows[0]["id"]
    with pytest.raises(SystemExit, match="frozen test split"):
        train_grpo.assert_no_test_overlap(rows, {leaked})


def test_dry_run_aborts_when_test_manifest_overlaps_train(tmp_path, monkeypatch):
    import train_sft

    from addr_repair.io import load_paired_records, records_hash, sha256_file

    cfg = _setup(tmp_path)
    resolved = train_sft.load_resolved_config(cfg)
    train_pairs, _ = train_sft.load_split_pairs(resolved, "train")
    rows, _ = train_grpo.select_grpo_rows(train_pairs, 2, seed=7)
    leaked = rows[0]["id"]
    raw_dir = tmp_path / "raw"
    records = load_paired_records(raw_dir / "dirty.csv", raw_dir / "clean.csv")
    by_id = {record["id"]: record for record in records}
    recs = [by_id[leaked]]
    (tmp_path / "manifests" / "test.json").write_text(
        json.dumps(
            {
                "entity_ids": [leaked],
                "records_sha256": records_hash(recs),
                "dirty_sha256": sha256_file(raw_dir / "dirty.csv"),
                "clean_sha256": sha256_file(raw_dir / "clean.csv"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["train_grpo.py", "--config", str(cfg), "--dry-run"])
    with pytest.raises(SystemExit, match="frozen test split"):
        train_grpo.main()
