from __future__ import annotations

import base64
import http.client
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from demo.session_recording import RecordingSnapshot  # noqa: E402
from demo.session_state import SessionState  # noqa: E402
from demo.web_session import MAX_BODY_BYTES, SessionHTTPServer  # noqa: E402
from onevoice.core.models.audio_chunk import AudioChunk  # noqa: E402
from onevoice.core.models.frame import Frame  # noqa: E402
from onevoice.core.models.speaker_track import SpeakerTrack  # noqa: E402


class FakeRunner:
    def __init__(self):
        self.live = False
        self.state = SessionState()
        self.record_busy = False
        self.busy = False
        self.starts = 0
        self.stops = 0
        self.closed = threading.Event()
        self.toggle_recording = Mock()
        self.recorder = SimpleNamespace(
            snapshot=lambda: RecordingSnapshot(False, 0.0, None, None)
        )
        self.tracks = [
            SpeakerTrack("a", (10, 20, 30, 40), 1.0, {"visible": True}),
            SpeakerTrack("b", (50, 20, 30, 40), 1.0, {"visible": True}),
            SpeakerTrack("c", (90, 20, 30, 40), 1.0, {"visible": False}),
        ]
        self.selector = SimpleNamespace(latest_tracks=lambda: self.tracks)
        self.toggle_captions = Mock()
        self._frame = None

    def frame(self):
        return self._frame

    def start(self):
        self.starts += 1
        self.busy = True
        self.state.begin_start()
        self.state.mark_running()
        self._frame = Frame(time.monotonic() * 1000.0, None, {})
        audio = AudioChunk(time.monotonic() * 1000.0, [0.0] * 32, 16000, 1, {})
        self.state.observe_frame(self._frame)
        self.state.observe_input(audio)
        self.state.observe_tracks(self.tracks)
        self.state.observe_output(audio, {"is_real_separation": False})

    def stop(self):
        self.stops += 1
        self.busy = False
        self.state.mark_stopped()

    def close(self, timeout=5.0):
        self.stop()
        self.closed.set()
        return True


@pytest.fixture
def local_server(tmp_path):
    (tmp_path / "index.html").write_text("<html>Local UI</html>", encoding="utf-8")
    (tmp_path / "app.css").write_text("body {}", encoding="utf-8")
    (tmp_path / "app.js").write_text("void 0;", encoding="utf-8")
    runner = FakeRunner()
    server = SessionHTTPServer(runner, ui_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, runner
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
        runner.close()


def request(server, path, *, method="GET", body=None, headers=None):
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def action(server, payload, **kwargs):
    return request(
        server,
        "/api/action",
        method="POST",
        body=json.dumps(payload),
        headers={
            "Content-Type": "application/json",
            "X-OneVoice-UI": "1",
            "Origin": server.origin,
            **kwargs,
        },
    )


def test_read_only_page_and_state_never_start_capture(local_server):
    server, runner = local_server
    status, headers, body = request(server, "/")
    assert status == 200 and b"Local UI" in body
    assert "no-store" in headers["Cache-Control"]
    assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    status, _, body = request(server, "/api/state")
    state = json.loads(body)
    assert status == 200
    assert state["session"]["phase"] == "ready"
    assert state["synthetic"] is True
    assert state["tracks"] == []
    assert state["frame"]["available"] is False
    assert state["recording"]["active"] is False
    assert state["server_time_ms"] > 0
    assert runner.starts == 0
    assert request(server, "/api/frame")[0] == 204


def test_start_select_clear_record_stop_use_existing_runner(local_server):
    server, runner = local_server
    status, _, body = action(server, {"action": "start"})
    assert status == 200 and runner.starts == 1
    state = json.loads(body)
    assert state["session"]["phase"] == "listening"
    assert len(state["tracks"]) == 3
    assert state["tracks"][1] == {
        "track_id": "b",
        "bounding_box": [50.0, 20.0, 30.0, 40.0],
        "visible": True,
        "person": "Person 2",
    }
    assert action(server, {"action": "select", "track_id": "b"})[0] == 200
    assert runner.state.selection()[0] == "b"
    assert action(server, {"action": "clear"})[0] == 200
    assert runner.state.selection()[0] is None
    assert action(server, {"action": "record"})[0] == 200
    runner.toggle_recording.assert_called_once()
    assert action(server, {"action": "captions"})[0] == 200
    runner.toggle_captions.assert_called_once()
    assert action(server, {"action": "stop"})[0] == 200
    assert runner.stops == 1
    assert runner.state.snapshot().phase == "stopped"


def test_captions_default_to_off_when_runner_has_no_captions_support(local_server):
    server, _ = local_server
    state = json.loads(request(server, "/api/state")[2])
    assert state["captions"] == {
        "enabled": False,
        "status": "off",
        "error": None,
        "current": None,
    }


def test_captions_snapshot_passes_through_when_runner_supports_it(local_server):
    server, runner = local_server
    runner.captions = SimpleNamespace(
        snapshot=lambda: {
            "enabled": True,
            "status": "ready",
            "error": None,
            "current": {"track_id": "a", "text": "hello", "timestamp_ms": 10.0},
        }
    )
    state = json.loads(request(server, "/api/state")[2])
    assert state["captions"]["enabled"] is True
    assert state["captions"]["current"]["track_id"] == "a"


@pytest.mark.parametrize("track_id", ["unknown", "c"])
def test_selection_rejects_disappeared_or_coasting_tracks(local_server, track_id):
    server, runner = local_server
    action(server, {"action": "start"})
    assert action(server, {"action": "select", "track_id": track_id})[0] == 409
    assert runner.state.selection()[0] is None


def test_cannot_select_before_start_or_after_stop(local_server):
    server, _ = local_server
    assert action(server, {"action": "select", "track_id": "a"})[0] == 409
    action(server, {"action": "start"})
    action(server, {"action": "stop"})
    assert action(server, {"action": "select", "track_id": "a"})[0] == 409


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "evil.example"},
        {"Host": "localhost:9999"},
        {"Origin": "https://evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
    ],
)
def test_other_origins_and_rebinding_hosts_cannot_read_state(local_server, headers):
    server, runner = local_server
    assert request(server, "/api/state", headers=headers)[0] == 403
    assert action(server, {"action": "start"}, **headers)[0] == 403
    assert runner.starts == 0


def test_missing_custom_header_and_preflight_cannot_start(local_server):
    server, runner = local_server
    assert (
        request(
            server,
            "/api/action",
            method="POST",
            body='{"action":"start"}',
            headers={"Content-Type": "application/json"},
        )[0]
        == 403
    )
    status, headers, _ = request(
        server,
        "/api/action",
        method="OPTIONS",
        headers={"Origin": "https://evil.example"},
    )
    assert status >= 400
    assert "Access-Control-Allow-Origin" not in headers
    assert runner.starts == 0


@pytest.mark.parametrize(
    ("body", "content_type", "expected"),
    [
        ('{"action":"start"}', "text/plain", 415),
        ("[1,2]", "application/json", 400),
        ("{broken", "application/json", 400),
        ("x" * (MAX_BODY_BYTES + 1), "application/json", 413),
    ],
)
def test_mutations_require_small_json_objects(
    local_server, body, content_type, expected
):
    server, runner = local_server
    assert (
        request(
            server,
            "/api/action",
            method="POST",
            body=body,
            headers={"X-OneVoice-UI": "1", "Content-Type": content_type},
        )[0]
        == expected
    )
    assert runner.starts == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "bogus"},
        {"action": "select"},
        {"action": "select", "track_id": 3},
        {"action": "select", "track_id": ""},
        {"action": "select", "track_id": "x" * 257},
    ],
)
def test_invalid_action_payloads_do_not_mutate(local_server, payload):
    server, runner = local_server
    assert action(server, payload)[0] == 400
    assert runner.starts == 0


@pytest.mark.parametrize("path", ["/../session_runtime.py", "/%2e%2e/secret", "/test"])
def test_static_whitelist_never_serves_other_files(local_server, path):
    server, _ = local_server
    assert request(server, path)[0] == 404
    assert request(server, "/app.css?v=1")[0] == 200
    assert request(server, "/app.js")[0] == 200


def test_frame_only_returns_real_active_pixels_and_stops_with_session(local_server):
    import cv2

    server, runner = local_server
    runner.start()
    runner._frame = Frame(
        time.monotonic() * 1000.0, np.full((24, 32, 3), 100, dtype=np.uint8), {}
    )

    assert request(server, "/api/frame")[0] == 204
    runner.live = True
    status, headers, image = request(server, "/api/frame?revision=1")
    assert status == 200 and headers["Content-Type"] == "image/jpeg"
    assert float(headers["X-Frame-Timestamp-Ms"]) == runner._frame.timestamp_ms
    assert cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).shape == (
        24,
        32,
        3,
    )
    frame = json.loads(request(server, "/api/state")[2])["frame"]
    assert frame["width"] == 32 and frame["height"] == 24
    runner.stop()
    assert request(server, "/api/frame")[0] == 204


def test_frame_header_matches_encoded_image_when_capture_advances(
    local_server, monkeypatch
):
    import cv2

    server, runner = local_server
    runner.start()
    runner.live = True
    first = Frame(1000.0, np.full((24, 32, 3), 50, dtype=np.uint8), {})
    second = Frame(2000.0, np.full((24, 32, 3), 180, dtype=np.uint8), {})
    runner._frame = first
    original_encode = cv2.imencode

    def advance_capture(*args, **kwargs):
        runner._frame = second
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(cv2, "imencode", advance_capture)
    status, headers, image = request(server, "/api/frame")
    assert status == 200
    assert float(headers["X-Frame-Timestamp-Ms"]) == first.timestamp_ms
    assert cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).mean() == 50

    _, headers, image = request(server, "/api/frame")
    assert float(headers["X-Frame-Timestamp-Ms"]) == second.timestamp_ms
    assert cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR).mean() == 180
    assert (
        request(server, "/api/frame")[1]["X-Frame-Timestamp-Ms"]
        == headers["X-Frame-Timestamp-Ms"]
    )


@pytest.mark.parametrize("phase", ["stopping", "interrupted"])
def test_stopping_or_interrupted_capture_has_no_available_frame(local_server, phase):
    server, runner = local_server
    runner.start()
    runner.live = True
    runner._frame = Frame(
        time.monotonic() * 1000.0, np.full((24, 32, 3), 100, dtype=np.uint8), {}
    )
    if phase == "stopping":
        runner.state.begin_stop()
    else:
        runner.state.clock = lambda: time.monotonic() + 3.0
    assert runner.state.snapshot().phase == phase
    state = json.loads(request(server, "/api/state")[2])
    assert state["frame"]["available"] is False
    status, headers, image = request(server, "/api/frame")
    assert status == 204 and not image
    assert "X-Frame-Timestamp-Ms" not in headers


def test_recording_paths_serialize_as_strings(local_server, tmp_path):
    server, runner = local_server
    clip = tmp_path / "clip"
    runner.recorder.snapshot = lambda: RecordingSnapshot(True, 1.2, clip, None)
    state = json.loads(request(server, "/api/state")[2])
    assert state["recording"]["path"] == str(clip)
    assert state["recording"]["elapsed_s"] == 1.2


def test_quit_acknowledges_and_releases_runner_off_request_thread(local_server):
    server, runner = local_server
    runner.start()
    status, _, body = request(
        server,
        "/api/quit",
        method="POST",
        body="{}",
        headers={"Content-Type": "application/json", "X-OneVoice-UI": "1"},
    )
    assert status == 200 and json.loads(body) == {"ok": True}
    assert runner.closed.wait(timeout=3)
    assert server.closing
    assert runner.state.snapshot().phase == "stopped"


def _post(server, path, payload, **extra):
    return request(
        server,
        path,
        method="POST",
        body=json.dumps(payload),
        headers={
            "Content-Type": "application/json",
            "X-OneVoice-UI": "1",
            "Origin": server.origin,
            **extra,
        },
    )


def test_summarize_endpoint_sends_only_the_server_side_transcript(
    local_server, monkeypatch
):
    server, runner = local_server
    runner.captions = SimpleNamespace(transcript_text=lambda: "Person: hello there")
    provider = Mock(return_value={"summary": "A greeting.", "key_points": []})
    monkeypatch.setattr("demo.summary.summarize", provider)
    status, _, _ = _post(server, "/api/summarize", {}, Origin="https://other.example")
    assert status == 403
    provider.assert_not_called()
    status, _, data = _post(server, "/api/summarize", {"text": "client text ignored"})
    assert status == 200 and json.loads(data)["summary"] == "A greeting."
    provider.assert_called_once_with("Person: hello there")
    assert runner.starts == 0
    with server.summary_lock:
        assert _post(server, "/api/summarize", {})[0] == 409


def test_summarize_needs_captions_and_maps_provider_errors(local_server, monkeypatch):
    server, runner = local_server
    status, _, data = _post(server, "/api/summarize", {})
    assert status == 409 and "No captions yet" in json.loads(data)["error"]
    runner.captions = SimpleNamespace(transcript_text=lambda: "   ")
    assert _post(server, "/api/summarize", {})[0] == 409
    runner.captions = SimpleNamespace(transcript_text=lambda: "Person: hi")
    monkeypatch.setattr(
        "demo.summary.summarize", Mock(side_effect=RuntimeError("Set the key"))
    )
    status, _, data = _post(server, "/api/summarize", {})
    assert status == 502 and "Set the key" in json.loads(data)["error"]


def test_speak_endpoint_returns_wav_and_rejects_foreign_origin(
    local_server, monkeypatch
):
    server, runner = local_server
    synth = Mock(return_value=b"\x00\x00" * 160)
    monkeypatch.setattr("demo.speech.synthesize", synth)
    status, _, _ = _post(
        server, "/api/speak", {"text": "hi"}, Origin="https://other.example"
    )
    assert status == 403
    synth.assert_not_called()
    status, _, data = _post(server, "/api/speak", {"text": "Connect the cable."})
    body = json.loads(data)
    assert status == 200 and body["sample_rate"] == 16000
    assert base64.b64decode(body["audio"])[:4] == b"RIFF"
    synth.assert_called_once_with("Connect the cable.")
    assert runner.starts == 0


def test_speak_endpoint_maps_errors_and_busy_state(local_server, monkeypatch):
    server, _ = local_server
    assert _post(server, "/api/speak", {"text": 5})[0] == 400
    monkeypatch.setattr(
        "demo.speech.synthesize", Mock(side_effect=ValueError("There is nothing"))
    )
    assert _post(server, "/api/speak", {"text": " "})[0] == 400
    monkeypatch.setattr(
        "demo.speech.synthesize", Mock(side_effect=RuntimeError("Set the key"))
    )
    status, _, data = _post(server, "/api/speak", {"text": "hi"})
    assert status == 502 and "Set the key" in json.loads(data)["error"]
    with server.speak_lock:
        assert _post(server, "/api/speak", {"text": "hi"})[0] == 409
