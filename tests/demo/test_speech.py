from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from demo import speech  # noqa: E402

FAKE_KEY = "test-key-not-real"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.fixture
def keyed(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", FAKE_KEY)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    monkeypatch.delenv("ONEVOICE_ELEVENLABS_MODEL", raising=False)


def _capture(monkeypatch, body=b"\x01\x00\x02\x00"):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["request"] = request
        seen["timeout"] = timeout
        return _Response(body)

    monkeypatch.setattr(speech.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_request_shape_and_pcm_returned(keyed, monkeypatch):
    seen = _capture(monkeypatch)
    pcm = speech.synthesize("Connect the blue cable.")
    request = seen["request"]
    assert pcm == b"\x01\x00\x02\x00"
    assert request.full_url == (
        f"{speech.API_ROOT}/{speech.DEFAULT_VOICE}?output_format=pcm_16000"
    )
    assert request.get_method() == "POST"
    assert request.get_header("Xi-api-key") == FAKE_KEY
    assert json.loads(request.data) == {
        "text": "Connect the blue cable.",
        "model_id": speech.DEFAULT_MODEL,
    }
    assert seen["timeout"] == speech.TIMEOUT_S


def test_voice_and_model_come_from_environment(keyed, monkeypatch):
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "voice123")
    monkeypatch.setenv("ONEVOICE_ELEVENLABS_MODEL", "model-x")
    seen = _capture(monkeypatch)
    speech.synthesize("hi")
    assert "/voice123?" in seen["request"].full_url
    assert json.loads(seen["request"].data)["model_id"] == "model-x"


def test_missing_key_never_reaches_the_network(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    def boom(*_a, **_k):
        raise AssertionError("network used without a key")

    monkeypatch.setattr(speech.urllib.request, "urlopen", boom)
    with pytest.raises(speech.SpeechError, match="ELEVENLABS_API_KEY"):
        speech.synthesize("hello")


@pytest.mark.parametrize("text", ["", "   ", "x" * (speech.MAX_CHARS + 1)])
def test_unusable_text_is_rejected_before_any_request(keyed, monkeypatch, text):
    monkeypatch.setattr(
        speech.urllib.request,
        "urlopen",
        lambda *_a, **_k: pytest.fail("request made for unusable text"),
    )
    with pytest.raises(speech.SpeechError):
        speech.synthesize(text)


@pytest.mark.parametrize(
    ("code", "fragment"),
    [(401, "rejected the API key"), (429, "quota"), (422, "voice, model or text")],
)
def test_http_errors_become_clear_messages_without_leaking_the_key(
    keyed, monkeypatch, code, fragment
):
    def fail(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, code, "err", {}, io.BytesIO(FAKE_KEY.encode())
        )

    monkeypatch.setattr(speech.urllib.request, "urlopen", fail)
    with pytest.raises(speech.SpeechError, match=fragment) as caught:
        speech.synthesize("hello")
    assert FAKE_KEY not in str(caught.value)


def test_network_failure_is_reported(keyed, monkeypatch):
    def fail(_request, timeout):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(speech.urllib.request, "urlopen", fail)
    with pytest.raises(speech.SpeechError, match="Could not reach"):
        speech.synthesize("hello")


def test_empty_audio_is_an_error_and_odd_byte_is_trimmed(keyed, monkeypatch):
    _capture(monkeypatch, body=b"")
    with pytest.raises(speech.SpeechError, match="no audio"):
        speech.synthesize("hello")
    _capture(monkeypatch, body=b"\x01\x00\x02")
    assert speech.synthesize("hello") == b"\x01\x00"


def test_cli_saves_audio_without_playing(keyed, monkeypatch, tmp_path, capsys):
    _capture(monkeypatch, body=b"\x00\x00" * 16000)
    target = tmp_path / "out.pcm"
    assert speech.main(["hello", "--save", str(target)]) == 0
    assert target.stat().st_size == 32000
    assert "1.0s" in capsys.readouterr().out


def test_cli_reports_failure_with_exit_code_1(monkeypatch, capsys):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert speech.main(["hello"]) == 1
    assert "ELEVENLABS_API_KEY" in capsys.readouterr().err
