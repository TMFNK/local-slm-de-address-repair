import csv
import json
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from addr_repair.io import load_paired_records, records_hash, sha256_file  # noqa: E402
from evaluate_local import main as evaluate_main  # noqa: E402


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
        "model_rev: rev-1\nprompt_rev: v1\nlocal_gguf: models/x.gguf\n"
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
    config_path.write_text("local_gguf: x\nserver_url: y\ndecode: {}\n", encoding="utf-8")

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
