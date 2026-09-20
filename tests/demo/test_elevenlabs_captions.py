import json
import threading

from demo.elevenlabs_transcription import ElevenLabsCaptions, RealtimeStream
from tests.demo.test_session_captions import _chunk, _ready_state


class FakeStream:
    def __init__(self, callback, status, settings):
        self.callback = callback
        self.status = status
        self.chunks = []
        self.stopped = False
        status("ready", None)

    def write(self, chunk):
        self.chunks.append(chunk)

    def stop(self):
        self.stopped = True


def setup():
    state = _ready_state()
    state.select("a")
    captions = ElevenLabsCaptions(state, stream_factory=FakeStream)
    captions.toggle()
    captions.on_output(_chunk(*state.selection()), True)
    return state, captions


def test_partial_replaces_text_and_only_final_enters_summary():
    _, captions = setup()
    stream = captions.stream
    stream.callback("hello", False)
    stream.callback("hello there", False)
    assert captions.snapshot()["current"]["text"] == "hello there"
    assert captions.transcript_text() == ""
    stream.callback("Hello there.", True)
    assert captions.transcript_text().count("Hello there.") == 1


def test_switch_rejects_old_results_and_old_audio():
    state, captions = setup()
    old = captions.stream
    old_chunk = _chunk(*state.selection())
    state.select("b")
    captions.on_output(old_chunk, True)
    old.callback("wrong speaker", True)
    assert old.stopped
    assert captions.stream is None
    assert not captions.transcript_text()
    captions.on_output(_chunk(*state.selection()), True)
    assert captions.stream is not old


def test_alarm_and_target_loss_do_not_send_audio():
    state, captions = setup()
    old = captions.stream
    captions.on_output(_chunk(*state.selection()), False)
    assert old.stopped
    assert len(old.chunks) == 1
    assert captions.snapshot()["current"] is None


def test_final_on_stop_is_saved_but_new_session_rejects_late_results():
    _, captions = setup()
    old = captions.stream
    captions.retire_current_sink()
    old.callback("Last sentence.", True)
    assert "Last sentence." in captions.transcript_text()
    captions.clear_transcript()
    old.callback("stale", True)
    assert not captions.transcript_text()


def test_errors_require_retry_and_ignore_old_connection_status():
    state, captions = setup()
    old = captions.stream
    old.status("error", "No credits")
    assert captions.snapshot()["error"] == "No credits"
    captions.toggle()
    captions.toggle()
    old.status("error", "stale")
    captions.on_output(_chunk(*state.selection()), True)
    assert captions.snapshot()["status"] == "ready"
    assert captions.snapshot()["error"] is None


def test_wire_format_and_final_commit(monkeypatch):
    import websocket

    sent = []
    received = threading.Event()

    class Socket:
        def settimeout(self, timeout):
            pass

        def send(self, payload):
            sent.append(json.loads(payload))
            received.set()

        def recv(self):
            raise websocket.WebSocketTimeoutException()

        def close(self):
            pass

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setattr(websocket, "create_connection", lambda *a, **kw: Socket())
    stream = RealtimeStream(lambda *args: None, lambda *args: None, {})
    stream.write(_chunk("a", 1))
    assert received.wait(2)
    stream.stop()
    stream.thread.join(3)
    assert not stream.thread.is_alive()
    assert sent[0]["message_type"] == "input_audio_chunk"
    assert sent[0]["sample_rate"] == 16000
    assert sent[-1]["commit"] is True


def test_missing_key_is_actionable(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    statuses = []
    stream = RealtimeStream(lambda *a: None, lambda *a: statuses.append(a), {})
    stream.thread.join(2)
    assert statuses[-1][0] == "error"
    assert ".env" in statuses[-1][1]
