"""Append-only JSONL audit log. One record per input."""

import hashlib
import json
import time
from pathlib import Path


def fingerprint_input(record: dict) -> str:
    blob = json.dumps(record, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def write_audit_record(
    path: str | Path,
    dirty: dict,
    output: dict,
    *,
    model_rev: str,
    prompt_rev: str,
    schema_ok: bool,
    latency_ms: float,
    score: dict | None = None,
) -> dict:
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_fingerprint": fingerprint_input(dirty),
        "model_rev": model_rev,
        "prompt_rev": prompt_rev,
        "dirty": dirty,
        "output": output,
        "changed_fields": [c["field"] for c in output.get("changes", [])],
        "needs_review": output.get("needs_review", []),
        "schema_ok": schema_ok,
        "latency_ms": round(latency_ms, 1),
        "score": score,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
