import base64
import io
import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from demo.visual_explain import Explanation, explain

IMAGE = base64.b64encode(b"\xff\xd8\xff" + b"0" * 30).decode()
RESULT = {
    "status": "found",
    "explanation": "The blue cable is on the left.",
    "objects": [{"label": "Blue cable", "box": [10, 20, 300, 400]}],
}


@pytest.mark.parametrize(
    "box",
    [
        [0, 0, 1001, 10],
        [20, 0, 10, 30],
        [0, 0, float("nan"), 10],
        [0, 0, 10],
        [0, -1, 10, 20],
    ],
)
def test_invalid_boxes_are_rejected(box):
    with pytest.raises(ValidationError):
        Explanation.model_validate({**RESULT, "objects": [{"label": "x", "box": box}]})


def test_missing_object_cannot_have_a_highlight():
    with pytest.raises(ValidationError):
        Explanation.model_validate({**RESULT, "status": "not_found"})


def test_provider_request_keeps_key_out_of_url_and_validates_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    response = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [{"text": json.dumps(RESULT)}],
                },
            }
        ]
    }
    with patch(
        "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(response).encode())
    ) as call:
        assert explain(IMAGE, "Show the blue cable") == RESULT
    request = call.call_args.args[0]
    assert "test-secret" not in request.full_url
    body = json.loads(request.data)
    assert body["contents"][0]["parts"][1]["inline_data"]["data"] == IMAGE


def test_missing_key_is_explicit(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        explain(IMAGE, "Show me that")


@pytest.mark.parametrize(
    "image,quote", [("bad", "hello"), (IMAGE, ""), (IMAGE, "x" * 1001)]
)
def test_invalid_input_never_calls_provider(image, quote):
    with patch("urllib.request.urlopen") as call:
        with pytest.raises(ValueError):
            explain(image, quote)
        call.assert_not_called()


def test_truncated_model_response_does_not_render(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    payload = {"candidates": [{"finishReason": "MAX_TOKENS"}]}
    with patch(
        "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())
    ):
        with pytest.raises(RuntimeError, match="no usable explanation"):
            explain(IMAGE, "Show me the cable")
