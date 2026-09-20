from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from demo import summary  # noqa: E402

FAKE_KEY = "gemini-test-key"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _reply(text, finish="STOP"):
    body = {
        "candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}]
    }
    return _Response(json.dumps(body).encode())


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("ONEVOICE_GEMINI_MODEL", raising=False)


def _serve(monkeypatch, response):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(summary.urllib.request, "urlopen", fake_urlopen)
    return seen


GOOD = json.dumps({"summary": "They planned lunch.", "key_points": ["Noon", "Cafe"]})


def test_request_carries_only_the_transcript_text(keyed, monkeypatch):
    seen = _serve(monkeypatch, _reply(GOOD))
    result = summary.summarize('Person: meet at noon "please"')
    request = seen["request"]
    assert result == {"summary": "They planned lunch.", "key_points": ["Noon", "Cafe"]}
    assert request.get_header("X-goog-api-key") == FAKE_KEY
    assert "gemini-2.5-flash:generateContent" in request.full_url
    body = json.loads(request.data)
    parts = body["contents"][0]["parts"]
    assert len(parts) == 1 and "inline_data" not in parts[0]
    assert '\\"please\\"' in parts[0]["text"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert seen["timeout"] == summary.TIMEOUT_S


def test_model_comes_from_environment_and_is_validated(keyed, monkeypatch):
    monkeypatch.setenv("ONEVOICE_GEMINI_MODEL", "gemini-x.1")
    seen = _serve(monkeypatch, _reply(GOOD))
    summary.summarize("Person: hi")
    assert "models/gemini-x.1:generateContent" in seen["request"].full_url
    monkeypatch.setenv("ONEVOICE_GEMINI_MODEL", "bad/model?x=1")
    with pytest.raises(RuntimeError, match="Invalid ONEVOICE_GEMINI_MODEL"):
        summary.summarize("Person: hi")


def test_missing_key_never_reaches_the_network(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        summary.urllib.request,
        "urlopen",
        lambda *_a, **_k: pytest.fail("network used without a key"),
    )
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        summary.summarize("Person: hi")


@pytest.mark.parametrize(
    "text", ["", "   ", None, "x" * (summary.MAX_TRANSCRIPT_CHARS + 1)]
)
def test_unusable_transcripts_are_rejected_before_any_request(keyed, monkeypatch, text):
    monkeypatch.setattr(
        summary.urllib.request,
        "urlopen",
        lambda *_a, **_k: pytest.fail("request made for an unusable transcript"),
    )
    with pytest.raises(ValueError):
        summary.summarize(text)


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        (401, "rejected"),
        (403, "access denied"),
        (429, "quota"),
        (500, "request failed"),
    ],
)
def test_http_errors_are_clear_and_never_leak_the_key(
    keyed, monkeypatch, code, fragment
):
    error = urllib.error.HTTPError("u", code, "e", {}, io.BytesIO(FAKE_KEY.encode()))
    _serve(monkeypatch, error)
    with pytest.raises(RuntimeError, match=fragment) as caught:
        summary.summarize("Person: hi")
    assert FAKE_KEY not in str(caught.value)


def test_network_failure_and_bad_responses_are_runtime_errors(keyed, monkeypatch):
    _serve(monkeypatch, urllib.error.URLError("no route"))
    with pytest.raises(RuntimeError, match="Could not reach"):
        summary.summarize("Person: hi")
    too_many = json.dumps({"summary": "ok", "key_points": list("123456")})
    for bad in (
        _reply(GOOD, finish="MAX_TOKENS"),
        _reply("not json"),
        _reply(json.dumps({"summary": "", "key_points": []})),
        _reply(too_many),
        _reply(json.dumps({"summary": "ok", "key_points": [], "extra": 1})),
        _Response(b"{}"),
    ):
        _serve(monkeypatch, bad)
        with pytest.raises(RuntimeError, match="unusable summary"):
            summary.summarize("Person: hi")
