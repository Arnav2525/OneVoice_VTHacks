

from __future__ import annotations

import array
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from demo.session_captions import SessionCaptions  # noqa: E402
from demo.session_state import SessionState  # noqa: E402
from demo.transcription import FakeTranscriber  # noqa: E402
from onevoice.core.models.audio_chunk import AudioChunk  # noqa: E402
from onevoice.core.models.speaker_track import SpeakerTrack  # noqa: E402

def _chunk(
    track_id: str, epoch: int, samples: list[float] | None = None
) -> AudioChunk:
    samples = samples if samples is not None else [0.2] * 16_000
    return AudioChunk(
        10_000.0,
        array.array("f", samples),
        16_000,
        1,
        {"target_track_id": track_id, "ui_selection_epoch": epoch},
    )

def _ready_state() -> SessionState:
    state = SessionState("live", clock=lambda: 10.0)
    state.begin_start()
    state.mark_running()
    state.observe_tracks(
        [
            SpeakerTrack("a", (0, 0, 10, 10), 1.0, {"visible": True}),
            SpeakerTrack("b", (0, 0, 10, 10), 1.0, {"visible": True}),
        ]
    )
    return state

def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False

def _flush_one_segment(captions: SessionCaptions, track_id: str, epoch: int) -> None:

    for _ in range(6):
        captions.on_output(_chunk(track_id, epoch), allowed=True)

def test_disabled_by_default_and_never_fed() -> None:
    state = _ready_state()
    captions = SessionCaptions(state, transcriber_factory=FakeTranscriber)
    assert captions.snapshot() == {
        "enabled": False,
        "status": "off",
        "error": None,
        "current": None,
    }
    assert state.select("a")
    captions.on_output(_chunk("a", state.selection()[1]), allowed=True)
    assert captions.snapshot()["current"] is None

def test_toggle_on_lazily_loads_once_and_reuses_across_toggles() -> None:
    load_calls = []

    def factory():
        load_calls.append(1)
        return FakeTranscriber(text="hello world")

    state = _ready_state()
    captions = SessionCaptions(state, transcriber_factory=factory)
    try:
        captions.toggle()
        assert captions.enabled
        assert _wait_for(lambda: captions.snapshot()["status"] == "ready")
        assert load_calls == [1]

        captions.toggle()
        assert not captions.enabled
        assert captions.snapshot()["status"] == "ready"

        captions.toggle()
        assert _wait_for(lambda: captions.snapshot()["status"] == "ready")
        assert load_calls == [1]
    finally:
        captions.close()

def test_load_failure_reports_error_and_disables() -> None:
    def factory():
        raise RuntimeError("no CUDA device")

    state = _ready_state()
    captions = SessionCaptions(state, transcriber_factory=factory)
    captions.toggle()
    assert _wait_for(lambda: captions.snapshot()["status"] == "error")
    snapshot = captions.snapshot()
    assert not snapshot["enabled"]
    assert "no CUDA device" in (snapshot["error"] or "")

def test_caption_appears_for_the_selected_speaker() -> None:
    state = _ready_state()
    captions = SessionCaptions(
        state, transcriber_factory=lambda: FakeTranscriber(text="hi there")
    )
    try:
        captions.toggle()
        assert _wait_for(lambda: captions.snapshot()["status"] == "ready")
        assert state.select("a")
        epoch = state.selection()[1]
        _flush_one_segment(captions, "a", epoch)
        assert _wait_for(lambda: captions.snapshot()["current"] is not None)
        current = captions.snapshot()["current"]
        assert current["track_id"] == "a"
        assert current["text"] == "hi there"
    finally:
        captions.close()

def test_muted_or_unselected_audio_is_never_fed_to_the_model() -> None:
    state = _ready_state()
    captions = SessionCaptions(
        state, transcriber_factory=lambda: FakeTranscriber(text="should not appear")
    )
    try:
        captions.toggle()
        assert _wait_for(lambda: captions.snapshot()["status"] == "ready")
        assert state.select("a")
        epoch = state.selection()[1]
        for _ in range(6):
            captions.on_output(_chunk("a", epoch), allowed=False)
        time.sleep(0.2)
        assert captions.snapshot()["current"] is None
    finally:
        captions.close()

def test_deselecting_clears_the_caption() -> None:
    state = _ready_state()
    captions = SessionCaptions(
        state, transcriber_factory=lambda: FakeTranscriber(text="hi there")
    )
    try:
        captions.toggle()
        assert _wait_for(lambda: captions.snapshot()["status"] == "ready")
        state.select("a")
        epoch = state.selection()[1]
        _flush_one_segment(captions, "a", epoch)
        assert _wait_for(lambda: captions.snapshot()["current"] is not None)

        state.select(None)
        captions.on_output(_chunk("a", epoch), allowed=False)
        assert captions.snapshot()["current"] is None
    finally:
        captions.close()

def test_switching_target_never_attributes_a_stale_transcript_to_the_new_person() -> (
    None
):

    release = threading.Event()

    class SlowTranscriber:
        def transcribe(self, audio, sample_rate) -> str:  # noqa: ANN001
            release.wait(timeout=2.0)
            return "old person's words"

    state = _ready_state()
    captions = SessionCaptions(state, transcriber_factory=SlowTranscriber)
    try:
        captions.toggle()
        assert _wait_for(lambda: captions.snapshot()["status"] == "ready")

        state.select("a")
        epoch_a = state.selection()[1]
        _flush_one_segment(captions, "a", epoch_a)

        state.select("b")
        epoch_b = state.selection()[1]
        captions.on_output(_chunk("b", epoch_b), allowed=True)

        release.set()
        time.sleep(0.3)
        assert captions.snapshot()["current"] is None
    finally:
        release.set()
        captions.close()

def test_retire_current_sink_stops_without_disabling() -> None:
    state = _ready_state()
    captions = SessionCaptions(
        state, transcriber_factory=lambda: FakeTranscriber(text="hi")
    )
    try:
        captions.toggle()
        assert _wait_for(lambda: captions.snapshot()["status"] == "ready")
        state.select("a")
        epoch = state.selection()[1]
        _flush_one_segment(captions, "a", epoch)
        assert _wait_for(lambda: captions.snapshot()["current"] is not None)

        captions.retire_current_sink()
        assert captions.snapshot()["current"] is None
        assert captions.enabled
    finally:
        captions.close()

def test_close_releases_the_model_and_resets_status() -> None:
    state = _ready_state()
    captions = SessionCaptions(state, transcriber_factory=FakeTranscriber)
    captions.toggle()
    assert _wait_for(lambda: captions.snapshot()["status"] == "ready")
    captions.close()
    assert captions.snapshot() == {
        "enabled": False,
        "status": "off",
        "error": None,
        "current": None,
    }
