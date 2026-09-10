import csv
import json
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import train_sft

from addr_repair.io import load_paired_records, records_hash, sha256_file
from addr_repair.prompts import build_repair_prompt
from addr_repair.targets import build_chat_messages

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def _row(entity_id, road="Musterstraße", postcode="80331", country="DE"):
    return {
        "id": entity_id,
        "name": "Example GmbH",
        "road": road,
        "house_number": "12",
        "postcode": postcode,
        "locality": "München",
        "country_code": country,
    }


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def _setup(tmp_path, train_rows, val_rows):
    """Write raw CSVs, manifests, and configs; return train config path."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_csv(raw_dir / "dirty.csv", [r[0] for r in train_rows + val_rows])
    _write_csv(raw_dir / "clean.csv", [r[1] for r in train_rows + val_rows])
    records = load_paired_records(raw_dir / "dirty.csv", raw_dir / "clean.csv")
    by_id = {r["id"]: r for r in records}
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()

    def _manifest(name, rows):
        recs = [by_id[r[0]["id"]] for r in rows]
        path = manifests_dir / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "entity_ids": [r["id"] for r in recs],
                    "records_sha256": records_hash(recs),
                    "dirty_sha256": sha256_file(raw_dir / "dirty.csv"),
                    "clean_sha256": sha256_file(raw_dir / "clean.csv"),
                }
            ),
            encoding="utf-8",
        )

    _manifest("train", train_rows)
    _manifest("val", val_rows)
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
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        f"base_config: {tmp_path / 'model.yaml'}\nlora_config: {tmp_path / 'lora.yaml'}\n"
        f"data_config: {tmp_path / 'data.yaml'}\noutput_dir: {tmp_path / 'checkpoints'}\n"
        "per_device_train_batch_size: 2\ngradient_accumulation_steps: 2\nlearning_rate: 0.0002\n"
        "num_train_epochs: 1\nbf16: false\nfp16: false\nmax_seq_length: 512\n"
        "eval_strategy: steps\neval_steps: 10\nsave_steps: 10\nsave_total_limit: 1\n"
        "logging_steps: 5\nreport_to: none\nseed: 7\ntrust_remote_code: true\n"
        "val_samples: 2\nval_gen_max_tokens: 64\n",
        encoding="utf-8",
    )
    return train_cfg


def _pair(entity_id, **dirty_overrides):
    dirty = _row(entity_id, **dirty_overrides)
    gold = _row(entity_id)
    return dirty, gold


def test_build_message_rows_abstain_and_validate():
    pairs = [
        {"id": "e1", "dirty": _row("e1", postcode=""), "gold": _row("e1")},
        {"id": "e2", "dirty": _row("e2", road="Musterstr."), "gold": _row("e2")},
    ]
    rows, stats = train_sft.build_message_rows(pairs)
    assert len(rows) == 2
    assert stats["invalid_targets"] == 0
    abstain = json.loads(rows[0]["messages"][-1]["content"])
    assert abstain["clean_record"]["postcode"] == ""
    assert "postcode" in abstain["needs_review"]
    # Gold holds '80331' for this row; the taught answer must not contain it.
    assert "80331" not in rows[0]["messages"][-1]["content"]


def test_build_message_rows_refuse_invalid_targets(monkeypatch):
    pairs = [{"id": "e1", "dirty": _row("e1"), "gold": _row("e1")}]
    monkeypatch.setattr(train_sft, "validate_output", lambda payload: (False, "boom"))
    with pytest.raises(SystemExit, match="fail validation"):
        train_sft.build_message_rows(pairs)


def test_prompt_parity_mismatch_refuses(tmp_path):
    cfg = _setup(tmp_path, [(_row("e1"), _row("e1"))], [(_row("e2"), _row("e2"))])
    resolved = train_sft.load_resolved_config(cfg)
    resolved["model"]["prompt_rev"] = "v99"
    with pytest.raises(SystemExit, match="drift"):
        train_sft.assert_prompt_parity(resolved)


def test_selection_score_prefers_honest_over_inventive():
    inventive = {
        "repair_precision": 0.39, "repair_recall": 1.0, "repair_f1": 0.56,
        "damage_rate": 0.05, "review_precision": 0.0,
        "contract_validity": 0.0, "inventions": 126,
    }
    honest = {
        "repair_precision": 1.0, "repair_recall": 0.08, "repair_f1": 0.14,
        "damage_rate": 0.0, "review_precision": 1.0,
        "contract_validity": 1.0, "inventions": 0,
    }
    assert train_sft.selection_score(honest) > train_sft.selection_score(inventive)
    name, _ = train_sft.pick_best([("ckpt-inventive", inventive), ("ckpt-honest", honest)])
    assert name == "ckpt-honest"


def test_pick_best_tie_breaks_by_contract_then_inventions():
    base = {
        "repair_precision": 0.5, "repair_recall": 0.5, "repair_f1": 0.5,
        "damage_rate": 0.0, "review_precision": 0.5,
    }
    a = {**base, "contract_validity": 0.5, "inventions": 5}
    b = {**base, "contract_validity": 0.9, "inventions": 5}
    c = {**base, "contract_validity": 0.9, "inventions": 1}
    assert train_sft.selection_score(a) == train_sft.selection_score(b)
    assert train_sft.pick_best([("a", a), ("b", b)])[0] == "b"
    assert train_sft.pick_best([("b", b), ("c", c)])[0] == "c"


def test_training_inference_prompt_parity():
    dirty, gold = _pair("e1", road="Musterstr.")
    assert build_chat_messages(dirty, gold)[:-1] == build_repair_prompt(dirty)


def test_dry_run_ok(tmp_path, monkeypatch, capsys):
    cfg = _setup(
        tmp_path,
        [(_row("e1", road="Musterstr."), _row("e1")), (_row("e2", postcode=""), _row("e2"))],
        [(_row("e3"), _row("e3"))],
    )
    monkeypatch.setattr(
        sys, "argv", ["train_sft.py", "--config", str(cfg), "--dry-run"]
    )
    with pytest.raises(SystemExit) as exc:
        train_sft.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "invalid=0" in out
    assert "mean(review_precision, 1-damage, F1)" in out


def test_dry_run_refuses_changed_inputs(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, [(_row("e1"), _row("e1"))], [(_row("e2"), _row("e2"))])
    manifest = json.loads((tmp_path / "manifests" / "train.json").read_text(encoding="utf-8"))
    manifest["records_sha256"] = "changed"
    (tmp_path / "manifests" / "train.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--config", str(cfg), "--dry-run"])
    with pytest.raises(SystemExit, match="differs from manifest"):
        train_sft.main()


def test_resume_flag_reaches_run_training(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, [(_row("e1"), _row("e1"))], [(_row("e2"), _row("e2"))])
    seen = {}
    monkeypatch.setattr(
        train_sft, "run_training", lambda resolved, resume=False: seen.update(resume=resume)
    )
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--config", str(cfg), "--resume"])
    train_sft.main()
    assert seen == {"resume": True}


def test_no_resume_by_default(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, [(_row("e1"), _row("e1"))], [(_row("e2"), _row("e2"))])
    seen = {}
    monkeypatch.setattr(
        train_sft, "run_training", lambda resolved, resume=False: seen.update(resume=resume)
    )
    monkeypatch.setattr(sys, "argv", ["train_sft.py", "--config", str(cfg)])
    train_sft.main()
    assert seen == {"resume": False}


def test_latest_checkpoint_picks_newest(tmp_path):
    assert train_sft._latest_checkpoint(tmp_path) is None
    old = tmp_path / "checkpoint-100"
    new = tmp_path / "checkpoint-200"
    old.mkdir()
    new.mkdir()
    import os
    import time as _time

    os.utime(old, (_time.time() - 10, _time.time() - 10))
    assert train_sft._latest_checkpoint(tmp_path) == new


def test_resolve_prefers_cwd_over_config_dir(tmp_path, monkeypatch):
    # Regression: /content/train_run.yaml must not redirect repo-relative
    # data paths to /content/ when a stray directory exists there.
    from pathlib import Path

    cwd_dir = tmp_path / "repo"
    cwd_dir.mkdir()
    (cwd_dir / "data").mkdir()
    (cwd_dir / "data" / "x.csv").write_text("real", encoding="utf-8")
    base_dir = tmp_path / "elsewhere"
    base_dir.mkdir()
    (base_dir / "data").mkdir()
    (base_dir / "data" / "x.csv").write_text("stray", encoding="utf-8")
    monkeypatch.chdir(cwd_dir)
    resolved = train_sft._resolve("data/x.csv", base_dir)
    assert resolved == Path.cwd() / "data" / "x.csv"
    assert resolved.read_text(encoding="utf-8") == "real"


def test_resolve_falls_back_to_config_dir(tmp_path, monkeypatch):
    from pathlib import Path

    cwd_dir = tmp_path / "repo"
    cwd_dir.mkdir()
    base_dir = tmp_path / "elsewhere"
    base_dir.mkdir()
    (base_dir / "data").mkdir()
    (base_dir / "data" / "x.csv").write_text("stray", encoding="utf-8")
    monkeypatch.chdir(cwd_dir)
    assert train_sft._resolve("data/x.csv", base_dir) == base_dir / "data" / "x.csv"
    assert train_sft._resolve(Path("/abs/y.csv"), base_dir) == Path("/abs/y.csv")


def test_training_template_has_generation_markers():
    # Without these, TRL cannot build the assistant-only loss mask and fails
    # at trainer init ("not training-compatible") — the Colab cell-9 error.
    assert "{%- generation %}" in train_sft.TRAIN_CHAT_TEMPLATE
    assert "{%- endgeneration %}" in train_sft.TRAIN_CHAT_TEMPLATE


def test_trl_recognises_training_template_markers():
    # The exact gate that failed Colab cell 9: TRL must see generation
    # markers, otherwise it tries (and fails) to patch the template.
    from trl.chat_template_utils import has_generation_markers

    assert has_generation_markers(train_sft.TRAIN_CHAT_TEMPLATE) is True


def test_training_template_is_no_think():
    assert "<think" not in train_sft.TRAIN_CHAT_TEMPLATE.lower()


def _render(messages, add_generation_prompt=False):
    import jinja2

    # Strip generation markers the way TRL does when rendering for masking.
    text = train_sft.TRAIN_CHAT_TEMPLATE.replace("{%- generation %}", "").replace(
        "{%- endgeneration %}", ""
    )
    return jinja2.Environment().from_string(text).render(
        messages=messages, add_generation_prompt=add_generation_prompt, bos_token="<s>"
    )


def test_training_template_renders_chatml_shapes():
    messages = [
        {"role": "system", "content": "Do repairs."},
        {"role": "user", "content": "Fix this."},
        {"role": "assistant", "content": '{"ok": true}'},
    ]
    text = _render(messages)
    assert "<|im_start|>system\nDo repairs.<|im_end|>" in text
    assert "<|im_start|>user\nFix this.<|im_end|>" in text
    assert "<|im_start|>assistant\n" in text
    assert '{"ok": true}' in text
    prompt_only = _render(messages[:2], add_generation_prompt=True)
    assert prompt_only.rstrip().endswith("<|im_start|>assistant")
    assert '{"ok": true}' not in prompt_only


def test_apply_training_template_swaps_and_returns_original():
    class StubTokenizer:
        chat_template = "original-template"

    tok = StubTokenizer()
    assert train_sft.apply_training_template(tok) == "original-template"
    assert tok.chat_template == train_sft.TRAIN_CHAT_TEMPLATE


def test_build_sft_config_constructs_real_config(tmp_path):
    # Guards the paid-GPU path: dry-run never builds SFTConfig, so an
    # unsupported kwarg (e.g. warmup_ratio on transformers 5.x) would only
    # explode on Colab. This test constructs it for real.
    cfg = _setup(tmp_path, [(_row("e1"), _row("e1"))], [(_row("e2"), _row("e2"))])
    resolved = train_sft.load_resolved_config(cfg)
    sft_args, dropped = train_sft.build_sft_config(
        resolved["train"], tmp_path / "out", use_bf16=False
    )
    assert dropped == []
    assert sft_args.assistant_only_loss is True
    assert sft_args.load_best_model_at_end is True
    assert sft_args.metric_for_best_model == "eval_loss"


def test_build_sft_config_reports_unsupported_kwargs(tmp_path, monkeypatch):
    from trl import SFTConfig

    cfg = _setup(tmp_path, [(_row("e1"), _row("e1"))], [(_row("e2"), _row("e2"))])
    resolved = train_sft.load_resolved_config(cfg)
    fields = dict(SFTConfig.__dataclass_fields__)
    removed = fields.pop("packing")
    monkeypatch.setattr(SFTConfig, "__dataclass_fields__", fields)
    try:
        _, dropped = train_sft.build_sft_config(resolved["train"], tmp_path / "out", False)
    finally:
        fields["packing"] = removed
    assert dropped == ["packing"]
