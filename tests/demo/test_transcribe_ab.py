

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo.level_meter import LevelMeterSink  # noqa: E402
from demo.tap_selection import TrackObservingSelector  # noqa: E402
from demo.transcribe_ab import build_ab_pipeline  # noqa: E402
from demo.transcription import FakeTranscriber  # noqa: E402
from onevoice.selection.first_track_selector import FirstTrackSelector  # noqa: E402

class _Collector:

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self._events = {"raw": threading.Event(), "isolated": threading.Event()}

    def __call__(self, label: str, text: str, timestamp_ms: float) -> None:
        self.calls.append((label, text, timestamp_ms))
        self._events[label].set()

    def wait_for(self, label: str, timeout: float = 5.0) -> bool:
        return self._events[label].wait(timeout)

def test_mock_mode_wires_both_taps_end_to_end() -> None:
    collector = _Collector()
    raw_transcriber = FakeTranscriber(text="raw heard something")
    isolated_transcriber = FakeTranscriber(text="isolated heard something")

    pipeline, mixture_sink, isolated_sink = build_ab_pipeline(
        config={},
        live=False,
        mixture_transcriber=raw_transcriber,
        isolated_transcriber=isolated_transcriber,
        max_segment_s=0.1,

        enable_vad=False,
    )

    mixture_sink._on_transcript = collector
    isolated_sink._on_transcript = collector

    pipeline.start()
    try:
        assert collector.wait_for("raw", timeout=10.0)
        assert collector.wait_for("isolated", timeout=10.0)
    finally:
        pipeline.stop()

    labels = {label for label, _, _ in collector.calls}
    assert labels == {"raw", "isolated"}
    texts = {label: text for label, text, _ in collector.calls}
    assert texts["raw"] == "raw heard something"
    assert texts["isolated"] == "isolated heard something"

def test_save_raw_too_without_save_dir_is_a_noop() -> None:

    pipeline, _mixture_sink, _isolated_sink = build_ab_pipeline(
        config={},
        live=False,
        save_dir=None,
        save_raw_too=True,
        mixture_transcriber=FakeTranscriber(),
        isolated_transcriber=FakeTranscriber(),
    )
    pipeline.start()
    pipeline.stop()

def test_display_mode_wiring_observes_tracks_and_meters_the_isolated_level() -> None:

    selector = TrackObservingSelector(FirstTrackSelector())
    level_meter = LevelMeterSink()

    pipeline, _mixture_sink, _isolated_sink = build_ab_pipeline(
        config={},
        live=False,
        mixture_transcriber=FakeTranscriber(),
        isolated_transcriber=FakeTranscriber(),
        target_selector=selector,
        level_meter=level_meter,
    )

    pipeline.start()
    try:

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not selector.latest_tracks():
            time.sleep(0.05)
    finally:
        pipeline.stop()

    assert (
        len(selector.latest_tracks()) >= 1
    )

    rms, peak = level_meter.snapshot()
    assert 0.0 <= rms <= 1.0
    assert 0.0 <= peak <= 1.0
