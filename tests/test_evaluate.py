import csv
import json
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from evaluate_local import main as evaluate_main

from addr_repair.io import load_paired_records, records_hash, sha256_file

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def _row(entity_id, road):
    return {
        "id": entity_id,
        "name": "Example GmbH",
        "road": road,
        "house_number": "12",
        "postcode": "80331",
        "locality": "München",
        "country_code": "DE",
    }


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_rules_writes_audit_metrics_and_freeze(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_csv(raw_dir / "dirty.csv", [_row("e1", "Musterstr."), _row("e2", "Musterstraße")])
    _write_csv(raw_dir / "clean.csv", [_row("e1", "Musterstraße"), _row("e2", "Musterstraße")])
    records = load_paired_records(raw_dir / "dirty.csv", raw_dir / "clean.csv")
    manifest_path = tmp_path / "test.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entity_ids": ["e1", "e2"],
                "records_sha256": records_hash(records),
                "dirty_sha256": sha256_file(raw_dir / "dirty.csv"),
                "clean_sha256": sha256_file(raw_dir / "clean.csv"),
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        "model_rev: rev-1\nprompt_rev: v2\nlocal_gguf: models/x.gguf\n"
        "server_url: http://127.0.0.1:8080\nn_ctx: 2048\nllama_cpp_rev: abc\n"
        "decode: {temperature: 0.0, top_p: 1.0, max_tokens: 512, seed: 7}\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "evals"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_local.py",
            "--system", "rules",
            "--manifest", str(manifest_path),
            "--raw-dir", str(raw_dir),
            "--out-dir", str(out_dir),
            "--config", str(config_path),
        ],
    )
    evaluate_main()

    system_dir = out_dir / "rules"
    audit = (system_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audit) == 2
    payload = json.loads((system_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["n"] == 2
    assert payload["metrics"]["repair_precision"] == 1.0
    assert payload["freeze"]["model_rev"] == "rules-floor-v1"
    assert payload["freeze"]["manifest_records_sha256"] == records_hash(records)
    freeze = json.loads((out_dir / "freeze.json").read_text(encoding="utf-8"))
    assert freeze["system"] == "rules"


def test_evaluate_refuses_changed_inputs(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_csv(raw_dir / "dirty.csv", [_row("e1", "Musterstr.")])
    _write_csv(raw_dir / "clean.csv", [_row("e1", "Musterstraße")])
    manifest_path = tmp_path / "test.json"
    manifest_path.write_text(
        json.dumps({"entity_ids": ["e1"], "records_sha256": "changed"}),
        encoding="utf-8",
    )
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        "local_gguf: x\nserver_url: y\ndecode: {}\nprompt_rev: v2\n", encoding="utf-8"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_local.py",
            "--system", "rules",
            "--manifest", str(manifest_path),
            "--raw-dir", str(raw_dir),
            "--out-dir", str(tmp_path / "evals"),
            "--config", str(config_path),
        ],
    )
    with pytest.raises(SystemExit, match="differs from manifest"):
        evaluate_main()


def _mock_config(tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        "model_rev: rev-1\nprompt_rev: v2\nlocal_gguf: models/x.gguf\n"
        "server_url: http://127.0.0.1:8080\nn_ctx: 2048\nllama_cpp_rev: abc\n"
        "decode: {temperature: 0.0, top_p: 1.0, max_tokens: 512, seed: 7}\n",
        encoding="utf-8",
    )
    return config_path


def _mock_setup(tmp_path):
    """Two-row manifest + raw + v2 config; returns (manifest, raw_dir, config)."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_csv(raw_dir / "dirty.csv", [_row("e1", "Musterstr."), _row("e2", "Musterstraße")])
    _write_csv(raw_dir / "clean.csv", [_row("e1", "Musterstraße"), _row("e2", "Musterstraße")])
    records = load_paired_records(raw_dir / "dirty.csv", raw_dir / "clean.csv")
    manifest_path = tmp_path / "test.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entity_ids": ["e1", "e2"],
                "records_sha256": records_hash(records),
                "dirty_sha256": sha256_file(raw_dir / "dirty.csv"),
                "clean_sha256": sha256_file(raw_dir / "clean.csv"),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, raw_dir, _mock_config(tmp_path)


def test_evaluate_base_mock_scores_validity_split_and_inventions(tmp_path, monkeypatch):
    # Fix 6: base path without a server — mocked client, real scoring/wiring.
    # e2 has an empty dirty postcode (gold filled); the fake fills it with a
    # made-up value, so the metrics must show an invention and a semantic miss.
    import evaluate_local

    from addr_repair.rules import repair_with_rules

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    dirty_rows = [_row("e1", "Musterstr."), {**_row("e2", "Musterstr."), "postcode": ""}]
    clean_rows = [_row("e1", "Musterstraße"), _row("e2", "Musterstraße")]
    _write_csv(raw_dir / "dirty.csv", dirty_rows)
    _write_csv(raw_dir / "clean.csv", clean_rows)
    records = load_paired_records(raw_dir / "dirty.csv", raw_dir / "clean.csv")
    manifest_path = tmp_path / "test.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entity_ids": ["e1", "e2"],
                "records_sha256": records_hash(records),
                "dirty_sha256": sha256_file(raw_dir / "dirty.csv"),
                "clean_sha256": sha256_file(raw_dir / "clean.csv"),
            }
        ),
        encoding="utf-8",
    )
    config_path = _mock_config(tmp_path)
    gguf_path = tmp_path / "fake.gguf"
    gguf_path.write_bytes(b"fake-gguf")

    class FakeModel:
        def __init__(self, **kwargs):
            pass

        def repair(self, dirty):
            repaired, changes, needs_review = repair_with_rules(dirty)
            # Invent a postcode fill to prove the wiring counts inventions.
            if not dirty.get("postcode", "").strip():
                repaired = dict(repaired)
                repaired["postcode"] = "99999"
                changes = list(changes) + [
                    {"field": "postcode", "from": "", "to": "99999"}
                ]
            return (
                {
                    "clean_record": repaired,
                    "changes": changes,
                    "needs_review": needs_review,
                },
                {
                    "model_rev": "fake-base",
                    "prompt_rev": "v2",
                    "latency_ms": 1.0,
                    "finish_reason": "stop",
                    "prompt_tokens": 0,
                    "predicted_tokens": 0,
                    "raw_text": "fake",
                    "parse_errors": [],
                },
            )

    monkeypatch.setattr(evaluate_local, "LocalModel", FakeModel)
    out_dir = tmp_path / "evals"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_local.py",
            "--system", "base",
            "--manifest", str(manifest_path),
            "--raw-dir", str(raw_dir),
            "--out-dir", str(out_dir),
            "--config", str(config_path),
            "--gguf", str(gguf_path),
        ],
    )
    evaluate_main()

    payload = json.loads((out_dir / "base" / "metrics.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    assert metrics["n"] == 2
    for key in ("semantic_validity", "parse_rate", "contract_validity", "inventions"):
        assert key in metrics
    assert metrics["parse_rate"] == 1.0
    # The fake invents e2's postcode: wiring must surface it as invention
    # with a semantic miss, and zero invented TP.
    assert metrics["inventions"] >= 1
    assert metrics["semantic_validity"] < 1.0
    assert metrics["by_field"]["postcode"]["correct_repairs"] == 0
    audits = (out_dir / "base" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(audits) == 2
    assert all("semantic_ok" in json.loads(line) for line in audits)
    assert payload["freeze"]["gguf_sha256"] is not None


def test_evaluate_sft_requires_model_rev(tmp_path, monkeypatch):
    manifest_path, raw_dir, config_path = _mock_setup(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_local.py",
            "--system", "sft",
            "--manifest", str(manifest_path),
            "--raw-dir", str(raw_dir),
            "--out-dir", str(tmp_path / "evals"),
            "--config", str(config_path),
        ],
    )
    with pytest.raises(SystemExit, match="model-rev"):
        evaluate_main()


def test_evaluate_refuses_prompt_rev_drift(tmp_path, monkeypatch):
    manifest_path, raw_dir, config_path = _mock_setup(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("prompt_rev: v2", "prompt_rev: v9"),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_local.py",
            "--system", "rules",
            "--manifest", str(manifest_path),
            "--raw-dir", str(raw_dir),
            "--out-dir", str(tmp_path / "evals"),
            "--config", str(config_path),
        ],
    )
    with pytest.raises(SystemExit, match="drift"):
        evaluate_main()
