"""Shared local inference path. Rules, base, and SFT score through one path.

Base and SFT are the same client against different GGUF files served by the
pinned llama.cpp server (started with `--reasoning off` for no-think mode).
Model quirks become parse errors in metadata, never exceptions; only a dead
server raises, so a typo in the server address fails fast instead of writing
thousands of error rows.
"""

import json
import time
import urllib.request

from addr_repair.parsing import parse_response
from addr_repair.prompts import PRODUCTION_PROMPT_REV, build_repair_prompt
from addr_repair.rules import repair_with_rules


class LocalModel:
    """Thin client for one GGUF behind the llama.cpp OpenAI endpoint."""

    def __init__(
        self,
        *,
        server_url: str,
        model: str,
        model_rev: str,
        decode: dict,
        timeout_s: int = 300,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.model = model
        self.model_rev = model_rev
        self.decode = decode
        self.timeout_s = timeout_s

    def repair(self, dirty: dict) -> tuple[dict | None, dict]:
        body = {
            "model": self.model,
            "messages": build_repair_prompt(dirty),
            "temperature": self.decode.get("temperature", 0.0),
            "top_p": self.decode.get("top_p", 1.0),
            "max_tokens": self.decode.get("max_tokens", 512),
            "seed": self.decode.get("seed", 7),
        }
        started = time.perf_counter()
        try:
            request = urllib.request.Request(
                f"{self.server_url}/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"llama.cpp server unreachable at {self.server_url}: {exc}. "
                "Start it first, e.g. llama-server -m <gguf> -c 2048 --reasoning off"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000
        message = payload["choices"][0]["message"]
        meta = {
            "model_rev": self.model_rev,
            "prompt_rev": PRODUCTION_PROMPT_REV,
            "latency_ms": round(latency_ms, 1),
            "finish_reason": payload["choices"][0].get("finish_reason"),
            "prompt_tokens": (payload.get("timings") or {}).get("prompt_n"),
            "predicted_tokens": (payload.get("timings") or {}).get("predicted_n"),
            "raw_text": message.get("content") or "",
        }
        output, parse_errors = parse_response(meta["raw_text"])
        meta["parse_errors"] = parse_errors
        return output, meta


def run_rules(dirty: dict) -> tuple[dict, dict]:
    """Rules floor through the same (output, meta) shape as LocalModel."""
    started = time.perf_counter()
    repaired, changes, needs_review = repair_with_rules(dirty)
    latency_ms = (time.perf_counter() - started) * 1000
    return (
        {"clean_record": repaired, "changes": changes, "needs_review": needs_review},
        {
            "model_rev": "rules-floor-v1",
            "prompt_rev": "n/a",
            "latency_ms": round(latency_ms, 1),
            "finish_reason": "rules",
            "prompt_tokens": None,
            "predicted_tokens": None,
            "raw_text": "",
            "parse_errors": [],
        },
    )
