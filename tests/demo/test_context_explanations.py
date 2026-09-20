import threading

from demo.context_explanations import ContextExplanations


def test_capture_uses_current_frame_and_partial_captions(tmp_path):
    import time
    from types import SimpleNamespace
    from unittest.mock import Mock

    import numpy as np

    from demo.context_explanations import capture_request

    service = Mock()
    runner = SimpleNamespace(
        live=True,
        recorder=SimpleNamespace(
            snapshot=lambda: SimpleNamespace(active=True, path=tmp_path)
        ),
        frame=lambda: SimpleNamespace(
            timestamp_ms=time.monotonic() * 1000,
            data=np.zeros((64, 64, 3), dtype=np.uint8),
        ),
        captions=SimpleNamespace(
            transcript_text=lambda limit: "Earlier sentence.",
            snapshot=lambda: {"current": {"text": "The blue cable", "final": False}},
        ),
        explanations=service,
    )
    capture_request(runner)
    path, image, text = service.request.call_args.args
    assert path == tmp_path
    assert image.startswith(b"\xff\xd8")
    assert "Earlier sentence." in text and "The blue cable" in text
    assert not list(tmp_path.iterdir())


def test_all_requests_hidden_until_stop_and_only_text_saved(tmp_path):
    service = ContextExplanations(provider=lambda image, text: "Meaning: " + text)
    service.request(tmp_path, b"image one", "first")
    service.request(tmp_path, b"image two", "second")
    service.executor.shutdown(wait=True)
    assert service.snapshot()["count"] == 2
    assert service.snapshot()["items"] == []
    service.reveal(tmp_path)
    assert [item["text"] for item in service.snapshot()["items"]] == [
        "Meaning: first",
        "Meaning: second",
    ]
    assert [path.name for path in tmp_path.iterdir()] == ["explanations.txt"]


def test_pending_reply_updates_original_folder_after_session_reset(tmp_path):
    done = threading.Event()

    def provider(image, text):
        done.wait(2)
        return "Original explanation"

    service = ContextExplanations(provider=provider)
    service.request(tmp_path, b"image", "text")
    service.reveal(tmp_path)
    service.reset()
    done.set()
    service.executor.shutdown(wait=True)
    assert service.snapshot()["items"] == []
    assert "Original explanation" in (tmp_path / "explanations.txt").read_text()


def test_failure_and_disabled_feature_do_not_affect_other_work(tmp_path):
    import pytest

    def fail(image, text):
        raise ValueError("API unavailable")

    service = ContextExplanations(provider=fail)
    service.request(tmp_path, b"image", "text")
    service.executor.shutdown(wait=True)
    service.reveal(tmp_path)
    assert service.snapshot()["items"][0]["error"] == "API unavailable"
    service.enabled = False
    with pytest.raises(ValueError, match="disabled"):
        service.request(tmp_path, b"image", "text")
