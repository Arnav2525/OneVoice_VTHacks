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


def _chunk(track_id: str, epoch: int, samples: list[float] | None = None) -> AudioChunk:
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


def test_resolve_prefers_large_v3_on_cuda_and_small_on_cpu(monkeypatch):
    import demo.session_captions as mod

    monkeypatch.setattr(mod, "_cuda_visible", lambda: True)
    assert mod.resolve_caption_settings(None)["model_size"] == "large-v3"
    monkeypatch.setattr(mod, "_cuda_visible", lambda: False)
    settings = mod.resolve_caption_settings({})
    assert (settings["device"], settings["model_size"]) == ("cpu", "small")


def test_gpu_probe_does_not_load_ctranslate_before_torch(monkeypatch):
    import builtins
    from types import SimpleNamespace

    import demo.session_captions as mod

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ctranslate2":
            raise AssertionError("Caption probe must not preload CTranslate2 cuDNN")
        if name == "torch":
            return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert mod._cuda_visible() is True


def test_explicit_captions_config_wins(monkeypatch):
    import demo.session_captions as mod

    monkeypatch.setattr(mod, "_cuda_visible", lambda: True)
    settings = mod.resolve_caption_settings(
        {"captions": {"device": "cpu", "model_size": "base", "compute_type": "float32"}}
    )
    assert settings == {
        "model_size": "base",
        "device": "cpu",
        "compute_type": "float32",
    }


class _CudaBrokenTranscriber:
    built: list = []

    def __init__(self, model_size, device, compute_type):
        self.device, self.model_size = device, model_size
        _CudaBrokenTranscriber.built.append((device, model_size))

    def transcribe(self, audio, sample_rate):
        if self.device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found")
        return ""


def test_visible_but_unusable_cuda_falls_back_to_cpu_small(monkeypatch):
    import demo.session_captions as mod
    import demo.transcription as transcription

    _CudaBrokenTranscriber.built.clear()
    monkeypatch.setattr(
        transcription, "FasterWhisperTranscriber", _CudaBrokenTranscriber
    )
    transcriber = mod._validated_transcriber(
        {"model_size": "large-v3", "device": "cuda", "compute_type": "int8"}
    )
    assert (transcriber.device, transcriber.model_size) == ("cpu", "small")
    assert _CudaBrokenTranscriber.built == [("cuda", "large-v3"), ("cpu", "small")]


def test_cpu_failure_is_not_swallowed(monkeypatch):
    import demo.session_captions as mod
    import demo.transcription as transcription

    class _Broken:
        def __init__(self, **_):
            raise RuntimeError("model file corrupt")

    monkeypatch.setattr(transcription, "FasterWhisperTranscriber", _Broken)
    try:
        mod._validated_transcriber(
            {"model_size": "small", "device": "cpu", "compute_type": "int8"}
        )
    except RuntimeError as exc:
        assert "corrupt" in str(exc)
    else:
        raise AssertionError("cpu failure was swallowed")


def _speak(captions: SessionCaptions, epoch: int, text: str) -> None:
    captions._on_transcript(str(epoch), text, 1.0)


def test_transcript_collects_only_the_selected_speakers_current_epoch() -> None:
    state = _ready_state()
    captions = SessionCaptions(state, transcriber_factory=FakeTranscriber)
    assert captions.transcript_text() == ""
    state.select("a")
    _, epoch = state.selection()
    _speak(captions, epoch, "  hello there ")
    _speak(captions, epoch - 1, "stale epoch line")
    _speak(captions, epoch, "   ")
    expected = f"{state.person_name('a')}: hello there"
    assert captions.transcript_text() == expected
    state.select(None)
    _speak(captions, epoch, "nobody is selected")
    assert captions.transcript_text() == expected


def test_transcript_is_bounded_and_keeps_the_newest_lines() -> None:
    state = _ready_state()
    captions = SessionCaptions(state, transcriber_factory=FakeTranscriber)
    state.select("a")
    _, epoch = state.selection()
    for number in range(400):
        _speak(captions, epoch, f"line {number}")
    lines = captions.transcript_text(max_chars=10_000_000).splitlines()
    assert len(lines) == 300 and lines[-1].endswith("line 399")
    short = captions.transcript_text(max_chars=60).splitlines()
    assert short and short[-1].endswith("line 399") and sum(map(len, short)) < 60


def test_transcript_clears_on_request_and_on_close() -> None:
    state = _ready_state()
    captions = SessionCaptions(state, transcriber_factory=FakeTranscriber)
    state.select("a")
    _, epoch = state.selection()
    _speak(captions, epoch, "keep me")
    captions.clear_transcript()
    assert captions.transcript_text() == ""
    _speak(captions, epoch, "again")
    captions.close()
    assert captions.transcript_text() == ""
