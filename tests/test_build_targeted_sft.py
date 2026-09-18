import json
import sys

import yaml

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

import build_targeted_sft

from addr_repair.io import load_paired_records, records_hash, sha256_file

FIELDS = ["name", "road", "house_number", "postcode", "locality", "country_code"]


def _write_csv(path, rows):
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)


def _row(entity_id, *, name="Example GmbH", road="Musterstraße", locality="München"):
    return {
        "id": entity_id,
        "name": name,
        "road": road,
        "house_number": "12",
        "postcode": "80331",
        "locality": locality,
        "country_code": "DE",
    }


def _setup(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    dirty = [
        _row("target-name", name="Example GmbH "),
        _row("target-locality", locality="Muenchen"),
        _row("control"),
        _row("other-only", name="Example GmbH", locality="München"),
        _row("test-row", name="Test dirty"),
    ]
    clean = [
        _row("target-name"),
        _row("target-locality"),
        _row("control"),
        _row("other-only", road="Andere Straße"),
        _row("test-row"),
    ]
    _write_csv(raw / "dirty.csv", dirty)
    _write_csv(raw / "clean.csv", clean)
    records = load_paired_records(raw / "dirty.csv", raw / "clean.csv")
    by_id = {record["id"]: record for record in records}
    manifests = tmp_path / "manifests"
    manifests.mkdir()

    def write_manifest(name, ids):
        pairs = [by_id[entity_id] for entity_id in ids]
        (manifests / f"{name}.json").write_text(
            json.dumps(
                {
                    "entity_ids": ids,
                    "records_sha256": records_hash(pairs),
                    "dirty_sha256": sha256_file(raw / "dirty.csv"),
                    "clean_sha256": sha256_file(raw / "clean.csv"),
                }
            ),
            encoding="utf-8",
        )

    write_manifest("train", ["target-name", "target-locality", "control", "other-only"])
    write_manifest("val", ["control"])
    write_manifest("test", ["test-row"])
    config = tmp_path / "data.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dirty_file": "dirty.csv",
                "clean_file": "clean.csv",
                "id_field": "id",
                "raw_dir": str(raw),
                "manifests_dir": str(manifests),
            }
        ),
        encoding="utf-8",
    )
    return config, manifests


def test_build_targeted_view_selects_target_rows_and_controls(tmp_path):
    config, manifests = _setup(tmp_path)
    out_dir = tmp_path / "targeted"

    result = build_targeted_sft.build_targeted_view(config, out_dir)

    assert result["target_rows"] == 2
    assert result["control_rows"] == 1
    assert result["train_rows"] == 3
    assert result["test_overlap"] == 0
    targeted = json.loads((out_dir / "train.json").read_text(encoding="utf-8"))
    assert targeted["entity_ids"] == ["target-name", "target-locality", "control"]
    assert targeted["selection"]["target_fields"] == ["name", "locality"]
    assert json.loads((out_dir / "val.json").read_text()) == json.loads(
        (manifests / "val.json").read_text()
    )
    assert json.loads((out_dir / "test.json").read_text()) == json.loads(
        (manifests / "test.json").read_text()
    )


def test_build_targeted_view_refuses_changed_source_manifest(tmp_path):
    config, manifests = _setup(tmp_path)
    manifest = json.loads((manifests / "train.json").read_text())
    manifest["records_sha256"] = "changed"
    (manifests / "train.json").write_text(json.dumps(manifest), encoding="utf-8")

    try:
        build_targeted_sft.build_targeted_view(config, tmp_path / "targeted")
    except SystemExit as exc:
        assert "train records hash differs" in str(exc)
    else:
        raise AssertionError("changed source manifest was accepted")
