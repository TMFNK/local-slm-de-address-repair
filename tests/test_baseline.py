import json
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from run_baseline import main as baseline_main

CLEAN = {
    "name": "Example GmbH",
    "road": "Musterstraße",
    "house_number": "12",
    "postcode": "80331",
    "locality": "München",
    "country_code": "DE",
}


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run(monkeypatch, tmp_path, *argv):
    monkeypatch.setattr(sys, "argv", ["run_baseline.py", *argv])
    monkeypatch.chdir(tmp_path)
    baseline_main()


def test_pairs_mode_writes_audit_and_metrics(tmp_path, monkeypatch):
    abbrev = dict(CLEAN, road="Musterstr.")
    missing = dict(CLEAN, postcode="", country_code="")
    pairs = tmp_path / "pairs.jsonl"
    _write_jsonl(
        pairs,
        [
            {"id": "e1", "dirty": abbrev, "gold": dict(CLEAN)},
            {"id": "e2", "dirty": dict(CLEAN), "gold": dict(CLEAN)},
            {"id": "e3", "dirty": missing, "gold": dict(CLEAN)},
        ],
    )
    out = tmp_path / "audit.jsonl"

    _run(monkeypatch, tmp_path, "--pairs", str(pairs), "--out", str(out))

    audit = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(audit) == 3
    assert all(entry["model_rev"] == "rules-floor-v1" for entry in audit)

    road_changes = [
        change
        for change in audit[0]["output"]["changes"]
        if change["field"] == "road"
    ]
    assert road_changes and road_changes[0]["from"] == "Musterstr."

    metrics_path = tmp_path / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["system"] == "rules-floor"
    metrics = payload["metrics"]
    assert metrics["n"] == 3
    assert metrics["repair_precision"] == 1.0
    assert metrics["repair_recall"] > 0
    assert set(metrics["by_field"]) == set(CLEAN)


def test_bare_mode_writes_audit_without_metrics(tmp_path, monkeypatch):
    fixture = tmp_path / "records.jsonl"
    _write_jsonl(fixture, [dict(CLEAN)])
    out = tmp_path / "audit.jsonl"

    _run(monkeypatch, tmp_path, "--fixture", str(fixture), "--out", str(out))

    assert len(out.read_text(encoding="utf-8").splitlines()) == 1
    assert not (tmp_path / "metrics.json").exists()


def test_pairs_and_fixture_are_mutually_exclusive(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run(
            monkeypatch,
            tmp_path,
            "--pairs",
            str(tmp_path / "pairs.jsonl"),
            "--fixture",
            str(tmp_path / "records.jsonl"),
        )
