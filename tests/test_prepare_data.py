import csv
import json
import sys

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import prepare_data

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def _row(entity_id):
    return {
        "id": entity_id,
        "name": "Example GmbH",
        "road": "Musterstraße",
        "house_number": "12",
        "postcode": "80331",
        "locality": "München",
        "country_code": "DE",
    }


def _setup(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        _row("e1"),
        _row("e2"),
        {**_row("e3"), "road": "Hauptstraße"},
        {**_row("e4"), "road": "Bahnhofstraße"},
    ]
    for name in ("dirty.csv", "clean.csv"):
        with (raw_dir / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", *FIELDS])
            writer.writeheader()
            writer.writerows(rows)
    (tmp_path / "data.yaml").write_text(
        "source_repo: https://example.invalid/repo\n"
        "zenodo_url: https://example.invalid/record\n"
        "dataset_date: '2026-06-26'\n"
        "license: ODbL-1.0\n"
        "dirty_file: dirty.csv\n"
        "clean_file: clean.csv\n"
        "id_field: id\n"
        "splits: {train_entities: 2, val_entities: 1, test_entities: 1}\n"
        f"smoke_fixture: {tmp_path / 'smoke'}/\n"
        f"manifests_dir: {tmp_path / 'manifests'}\n",
        encoding="utf-8",
    )
    return tmp_path / "data.yaml"


def test_skip_archive_allows_subset_fetch(tmp_path, monkeypatch):
    config = _setup(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data.py", "--config", str(config), "--raw-dir", str(tmp_path / "raw"),
         "--skip-archive"],
    )
    prepare_data.main()
    manifest = json.loads((tmp_path / "manifests" / "train.json").read_text(encoding="utf-8"))
    assert manifest["archive"] is None
    assert manifest["archive_sha256"] is None
    assert manifest["archive_skipped"] is True
    assert manifest["entity_count"] == 2

    manifests = {
        name: json.loads((tmp_path / "manifests" / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("train", "val", "test")
    }
    smoke_manifest = json.loads(
        (tmp_path / "smoke" / "manifest.json").read_text(encoding="utf-8")
    )
    smoke_pairs = [
        json.loads(line)
        for line in (tmp_path / "smoke" / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    smoke_records = [
        json.loads(line)
        for line in (tmp_path / "smoke" / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assignments = {
        entity_id: name
        for name, manifest in manifests.items()
        for entity_id in manifest["entity_ids"]
    }
    assert assignments["e1"] == assignments["e2"]
    assert smoke_manifest["split"] == "train"
    assert smoke_manifest["test_is_original_pairs"] is False
    assert set(smoke_manifest["entity_ids"]) <= set(manifests["train"]["entity_ids"])
    assert not set(smoke_manifest["entity_ids"]) & set(manifests["test"]["entity_ids"])
    assert smoke_records == [record["dirty"] for record in smoke_pairs]


def test_missing_archive_refuses_without_flag(tmp_path, monkeypatch):
    _setup(tmp_path)
    (tmp_path / "data_with_archive.yaml").write_text(
        (tmp_path / "data.yaml").read_text(encoding="utf-8") + "archive_file: AddressTable.zip\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_data.py", "--config", str(tmp_path / "data_with_archive.yaml"),
         "--raw-dir", str(tmp_path / "raw")],
    )
    with pytest.raises(SystemExit, match="not available"):
        prepare_data.main()
