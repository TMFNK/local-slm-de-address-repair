"""Dataset manifests, hashes, split helpers. No raw data in git."""

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def entity_split(ids: list[str], train: int, val: int, test: int, seed: int = 7) -> dict:
    """Deterministic split by sorted id. Caller guarantees no id overlap."""
    ordered = sorted(set(ids))
    return {
        "train": ordered[:train],
        "val": ordered[train : train + val],
        "test": ordered[train + val : train + val + test],
    }
