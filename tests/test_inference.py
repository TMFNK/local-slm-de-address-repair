import json
import sys
import urllib.request

import pytest

sys.path.insert(0, "src")

from addr_repair.inference import LocalModel, run_rules
from addr_repair.prompts import PRODUCTION_PROMPT_REV

CLEAN = {
    "name": "Example GmbH",
    "road": "Musterstraße",
    "house_number": "12",
    "postcode": "80331",
    "locality": "München",
    "country_code": "DE",
}

DECODE = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": 7}


def _client(monkeypatch, content, **timings):
    payload = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "timings": {"prompt_n": 10, "predicted_n": 5, **timings},
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = LocalModel(
        server_url="http://127.0.0.1:8080",
        model="test.gguf",
        model_rev="rev-1",
        decode=DECODE,
    )
    return client, seen


def test_local_model_returns_parsed_output(monkeypatch):
    text = json.dumps({"clean_record": CLEAN, "changes": [], "needs_review": []})
    client, seen = _client(monkeypatch, text)

    output, meta = client.repair(dict(CLEAN, road="Musterstr."))

    assert output["clean_record"] == CLEAN
    assert meta["parse_errors"] == []
    assert meta["model_rev"] == "rev-1"
    assert meta["prompt_rev"] == PRODUCTION_PROMPT_REV
    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert seen["body"]["seed"] == 7
    assert seen["body"]["temperature"] == 0.0


def test_local_model_reports_parse_errors(monkeypatch):
    text = '```json\n{"clean_record": "flat", "changes": [], "needs_review": []}\n```'
    client, _ = _client(monkeypatch, text)

    output, meta = client.repair(dict(CLEAN))

    assert output is None
    assert meta["parse_errors"] == ["clean_record-is-not-an-object"]


def test_local_model_raises_helpfully_when_server_down(monkeypatch):
    def dead(request, timeout=None):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(urllib.request, "urlopen", dead)
    client = LocalModel(
        server_url="http://127.0.0.1:9999", model="x", model_rev="r", decode=DECODE
    )

    with pytest.raises(RuntimeError, match="--reasoning off"):
        client.repair(dict(CLEAN))


def test_run_rules_matches_client_shape():
    output, meta = run_rules(dict(CLEAN, road="Musterstr."))

    assert set(output) == {"clean_record", "changes", "needs_review"}
    assert meta["parse_errors"] == []
    assert meta["model_rev"] == "rules-floor-v1"
    assert output["clean_record"]["road"] == "Musterstraße"
