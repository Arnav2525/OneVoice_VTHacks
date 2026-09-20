from __future__ import annotations

import array
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from demo import session_runtime as runtime  # noqa: E402
from demo.safety import SafetyEvent, build_safety_monitor  # noqa: E402
from demo.session_state import SessionState  # noqa: E402
from onevoice.core.models.audio_chunk import AudioChunk  # noqa: E402
from onevoice.core.models.frame import Frame  # noqa: E402
from onevoice.core.models.speaker_track import SpeakerTrack  # noqa: E402


def audio(value=0.25, timestamp_ms=10000.0, **metadata):
    return AudioChunk(timestamp_ms, array.array("f", [value] * 32), 16000, 1, metadata)


def track(name):
    return SpeakerTrack(name, (0, 0, 100, 100), 1.0, {"visible": True})


class FakeMonitor:
    """Stands in for SafetyMonitor without a classifier or worker thread."""

    def __init__(self, active=False, raw=None):
        self._active = active
        self._raw = raw
        self.writes = []
        self.started = False
        self.stopped = False

    def write(self, chunk):
        self.writes.append(chunk)

    def is_active(self):
        return self._active

    def latest_raw_chunk(self):
        return self._raw

    def set(self, active, raw=None):
        self._active = active
        if raw is not None:
            self._raw = raw

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def prepared_sink(safety=None, mode="live"):
    """A sink with a live, selected, fully-eligible session behind it."""
    state = SessionState(mode, clock=lambda: 10.0)
    state.begin_start()
    state.mark_running()
    state.observe_input(audio())
    state.observe_frame(Frame(10000, [[[0, 0, 0]]], {}))
    state.observe_tracks([track("a"), track("b")])
    assert state.select("a")
    status = {"is_real_separation": True, "fallback_active": False}
    destination, recorded, captioned = [], [], []
    sink = runtime.SessionSink(
        SimpleNamespace(write=destination.append),
        state,
        SimpleNamespace(get_status=lambda: status),
        SimpleNamespace(on_output=recorded.append),
        SimpleNamespace(
            on_output=lambda chunk, allowed: captioned.append((chunk, allowed))
        ),
        safety=safety,
    )
    eligible = audio(
        target_track_id="a",
        conditioning="visual",
        output_ready=True,
        ui_selection_epoch=state.selection()[1],
        fallback=False,
    )
    return sink, state, eligible, destination, recorded, captioned


# --- passthrough while the alarm sounds -------------------------------------


def test_alarm_passes_raw_audio_through_instead_of_silence():
    """Without safety this chunk is muted to zeros; the alarm must override that."""
    raw = audio(value=0.8, room_audio=True)
    monitor = FakeMonitor(active=True, raw=raw)
    sink, _, _, destination, _, _ = prepared_sink(safety=monitor)

    # An ineligible chunk: no target metadata, so output_allowed() is False.
    sink.write(audio(value=0.0))

    assert len(destination) == 1
    assert any(destination[0].data), "alarm audio must not be muted to silence"
    assert destination[0].metadata.get("room_audio") is True


def test_alarm_audio_is_marked_as_a_safety_override():
    raw = audio(value=0.8)
    sink, _, _, destination, _, _ = prepared_sink(
        safety=FakeMonitor(active=True, raw=raw)
    )

    sink.write(audio(value=0.0))

    assert destination[0].metadata["safety_override"] is True


def test_alarm_ramps_in_rather_than_clicking():
    raw = audio(value=1.0)
    sink, _, _, destination, _, _ = prepared_sink(
        safety=FakeMonitor(active=True, raw=raw)
    )

    sink.write(audio(value=0.0))

    assert destination[0].data[0] < 0.2, "override must fade in from silence"
    assert destination[0].data[-1] > 0.8, "override must reach full gain"


def test_sustained_alarm_plays_at_full_gain_after_the_first_chunk():
    monitor = FakeMonitor(active=True, raw=audio(value=1.0))
    sink, _, _, destination, _, _ = prepared_sink(safety=monitor)

    sink.write(audio(value=0.0))
    monitor.set(True, audio(value=1.0, timestamp_ms=10020.0))
    sink.write(audio(value=0.0, timestamp_ms=10020.0))

    assert all(v > 0.99 for v in destination[1].data), "no ramp on continued alarm"


def test_alarm_bypasses_the_denoiser_and_rumble_filter():
    """Processing is tuned for speech; it must not touch emergency audio."""
    calls = []
    processor = SimpleNamespace(
        process=lambda chunk: calls.append(chunk) or chunk,
        reset=lambda: None,
    )
    sink, _, _, _, _, _ = prepared_sink(safety=FakeMonitor(active=True, raw=audio(0.8)))
    sink.cleanup = processor
    sink.denoiser = processor

    sink.write(audio(value=0.0))

    assert calls == [], "alarm audio must reach the ear unprocessed"


def test_alarm_keeps_the_playback_clock_fresh():
    """Otherwise the UI reports 'Audio output interrupted' during the alarm."""
    sink, state, _, _, _, _ = prepared_sink(
        safety=FakeMonitor(active=True, raw=audio(0.8))
    )

    sink.write(audio(value=0.0))

    assert state._playback_at is not None


def test_alarm_audio_is_recorded():
    sink, _, _, destination, recorded, _ = prepared_sink(
        safety=FakeMonitor(active=True, raw=audio(0.8))
    )

    sink.write(audio(value=0.0))

    assert recorded == destination


def test_room_audio_is_not_captioned_as_the_selected_person():
    sink, _, _, _, _, captioned = prepared_sink(
        safety=FakeMonitor(active=True, raw=audio(0.8))
    )

    sink.write(audio(value=0.0))

    assert captioned and captioned[0][1] is False


def test_alarm_falls_back_to_the_pipeline_chunk_when_no_raw_audio_is_buffered():
    sink, _, _, destination, _, _ = prepared_sink(
        safety=FakeMonitor(active=True, raw=None)
    )

    sink.write(audio(value=0.5))

    assert len(destination) == 1
    assert destination[0].metadata["safety_override"] is True


# --- the quiet path must be untouched ---------------------------------------


def test_inactive_monitor_leaves_muting_identical_to_today():
    with_safety, _, chunk, destination, _, _ = prepared_sink(safety=FakeMonitor(False))
    without, _, chunk2, baseline, _, _ = prepared_sink(safety=None)

    with_safety.write(chunk)
    without.write(chunk2)

    assert list(destination[0].data) == list(baseline[0].data)
    assert destination[0].metadata == baseline[0].metadata


def test_no_monitor_configured_behaves_exactly_as_before():
    sink, _, eligible, destination, _, _ = prepared_sink(safety=None)

    sink.write(eligible)

    assert any(destination[0].data), "eligible audio still plays"
    assert "safety_override" not in destination[0].metadata


def test_eligible_audio_still_plays_when_no_alarm_is_sounding():
    sink, _, eligible, destination, _, _ = prepared_sink(safety=FakeMonitor(False))

    sink.write(eligible)

    assert any(destination[0].data)


def test_isolation_is_muted_again_once_the_alarm_releases():
    """Release fades out; it must not leave raw room audio playing."""
    monitor = FakeMonitor(active=True, raw=audio(1.0))
    sink, _, _, destination, _, _ = prepared_sink(safety=monitor)

    sink.write(audio(value=0.0))
    monitor.set(False)
    sink.write(audio(value=0.0, timestamp_ms=10020.0))

    assert destination[1].data[-1] == 0.0, "must fade back out on release"


# --- stop on release, not on activation -------------------------------------


class FakePipeline:
    def __init__(self):
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.is_running = False

    def start(self):
        self.is_running = True
        self.started.set()

    def stop(self):
        self.is_running = False
        self.stopped.set()


class FakeSeparator:
    def __init__(self):
        self.closed = threading.Event()

    def close(self):
        self.closed.set()


def running_runner():
    pipeline = FakePipeline()
    runner = runtime.SessionRunner({}, factory=lambda: (pipeline, FakeSeparator()))
    runner.start()
    assert pipeline.started.wait(timeout=5), "session never started"
    return runner, pipeline


def test_alarm_activation_does_not_stop_the_session():
    """Stopping mid-alarm would silence the earbuds exactly when it matters."""
    runner, pipeline = running_runner()
    try:
        runner._on_safety_change(True, SafetyEvent("Smoke detector, smoke alarm", 0.9))

        assert not pipeline.stopped.wait(timeout=0.3), "must keep playing the alarm"
    finally:
        assert runner.close(timeout=5)


def test_session_stops_once_the_alarm_clears():
    runner, pipeline = running_runner()
    try:
        runner._on_safety_change(True, SafetyEvent("Smoke detector, smoke alarm", 0.9))
        runner._on_safety_change(False, None)

        assert pipeline.stopped.wait(timeout=5), "session must end after the alarm"
    finally:
        assert runner.close(timeout=5)


def test_the_stop_explains_that_an_alarm_caused_it():
    runner, _ = running_runner()
    runner._on_safety_change(True, SafetyEvent("Smoke detector, smoke alarm", 0.9))
    runner._on_safety_change(False, None)
    assert runner.close(timeout=5)

    view = runner.state.snapshot()
    assert view.phase == "stopped", "an alarm is not a crash"
    assert "alarm" in view.detail.lower()
    assert "Smoke detector, smoke alarm" in view.detail


def test_an_ordinary_stop_does_not_mention_an_alarm():
    runner, _ = running_runner()
    assert runner.close(timeout=5)

    assert "alarm" not in runner.state.snapshot().detail.lower()


def test_preview_sessions_do_not_build_an_alarm_monitor(monkeypatch):
    """Preview plays through a mock sink, so there is nothing to hear."""
    built = []
    monkeypatch.setattr(
        runtime,
        "build_safety_monitor",
        lambda *a, **k: built.append(k) or FakeMonitor(),
    )
    runner = runtime.SessionRunner({}, live=False)
    runner._build()

    assert built == [], "preview must not load an alarm detector"
    assert runner._safety is None


def test_the_monitor_hears_the_microphone_before_isolation(monkeypatch):
    """The detector must tap raw input, not the separated output."""
    monitor = FakeMonitor()
    monkeypatch.setattr(runtime, "build_safety_monitor", lambda *a, **k: monitor)
    runner = runtime.SessionRunner({}, live=False)
    runner.preview = False

    pipeline, _ = runner._build()
    pipeline._audio_source.start()
    try:
        chunk = pipeline._audio_source.read()
    finally:
        pipeline._audio_source.stop()

    assert monitor.writes == [chunk], "monitor must see raw microphone audio"
    assert pipeline._audio_sink.safety is monitor, "sink must consult the monitor"


def test_the_monitor_starts_and_stops_with_the_audio_source(monkeypatch):
    monitor = FakeMonitor()
    monkeypatch.setattr(runtime, "build_safety_monitor", lambda *a, **k: monitor)
    runner = runtime.SessionRunner({}, live=False)
    runner.preview = False

    pipeline, _ = runner._build()
    pipeline._audio_source.start()
    assert monitor.started, "detector thread must run with the session"
    pipeline._audio_source.stop()

    assert monitor.stopped, "detector thread must not outlive the session"


def test_a_real_failure_still_reports_as_an_error():
    state = SessionState("live")
    state.begin_start()
    state.mark_stopped("the camera fell over")

    view = state.snapshot()
    assert view.phase == "error"
    assert view.detail == "the camera fell over"


def test_an_ordinary_stop_still_reads_as_a_normal_stop():
    state = SessionState("live")
    state.begin_start()
    state.mark_stopped()

    view = state.snapshot()
    assert view.phase == "stopped"
    assert "released" in view.detail


# --- building the monitor from config ---------------------------------------


def test_safety_is_built_when_enabled():
    monitor = build_safety_monitor(
        {"safety": {"enabled": True}},
        sample_rate=16000,
        classifier_factory=lambda classes: SimpleNamespace(classify=lambda a, r: None),
    )
    assert monitor is not None


def test_safety_is_absent_when_disabled():
    monitor = build_safety_monitor(
        {"safety": {"enabled": False}},
        sample_rate=16000,
        classifier_factory=lambda classes: SimpleNamespace(classify=lambda a, r: None),
    )
    assert monitor is None


def test_safety_is_on_by_default():
    """A safety feature that is off unless asked for is the wrong default."""
    monitor = build_safety_monitor(
        {},
        sample_rate=16000,
        classifier_factory=lambda classes: SimpleNamespace(classify=lambda a, r: None),
    )
    assert monitor is not None


def test_a_missing_classifier_disables_safety_instead_of_blocking_the_session(caplog):
    def explode(classes):
        raise ImportError("No module named 'tensorflow_hub'")

    with caplog.at_level(logging.WARNING):
        monitor = build_safety_monitor(
            {"safety": {"enabled": True}},
            sample_rate=16000,
            classifier_factory=explode,
        )

    assert monitor is None
    assert "safety" in caplog.text.lower()


def test_non_16k_audio_disables_safety_loudly(caplog):
    """YAMNet raises on non-16k, and the monitor would swallow that forever."""
    with caplog.at_level(logging.WARNING):
        monitor = build_safety_monitor(
            {"safety": {"enabled": True}},
            sample_rate=48000,
            classifier_factory=lambda classes: SimpleNamespace(
                classify=lambda a, r: None
            ),
        )

    assert monitor is None
    assert "16" in caplog.text


def test_monitored_classes_and_thresholds_come_from_config():
    seen = {}

    def factory(classes):
        seen["classes"] = classes
        return SimpleNamespace(classify=lambda a, r: None)

    monitor = build_safety_monitor(
        {
            "safety": {
                "monitored_classes": ["Siren"],
                "activate_thresh": 0.75,
                "release_hold_s": 9.0,
            }
        },
        sample_rate=16000,
        classifier_factory=factory,
    )

    assert seen["classes"] == {"Siren"}
    assert monitor._activate_thresh == 0.75
    assert monitor._release_hold_s == 9.0


# --- end-to-end through the real monitor ------------------------------------


def test_a_detected_alarm_reaches_the_ear_and_then_stops_the_session():
    """The whole feature, driven by a real SafetyMonitor."""
    from demo.safety import FakeClassifier, SafetyMonitor

    stopped = threading.Event()
    runner = runtime.SessionRunner({})
    runner.stop = lambda: stopped.set()

    classifier = FakeClassifier(
        script=[SafetyEvent("Smoke detector, smoke alarm", 0.9)]
    )
    monitor = SafetyMonitor(
        classifier,
        monitored_classes={"Smoke detector, smoke alarm"},
        window_s=0.002,
        hop_s=0.002,
        release_hold_s=0.05,
        sample_rate=16000,
        on_state_change=runner._on_safety_change,
    )
    monitor.start()
    try:
        monitor.write(audio(value=0.8, timestamp_ms=10000.0))
        assert classifier.wait_for_call(timeout=2.0)
        for _ in range(200):
            if monitor.is_active():
                break
            threading.Event().wait(0.01)
        assert monitor.is_active(), "alarm should have activated the override"

        sink, _, _, destination, _, _ = prepared_sink(safety=monitor)
        sink.write(audio(value=0.0))
        assert any(destination[0].data), "listener must hear the alarm"

        monitor.write(audio(value=0.0, timestamp_ms=11000.0))
        assert stopped.wait(timeout=2.0), "session must stop once the alarm clears"
    finally:
        monitor.stop()


@pytest.mark.parametrize("class_name", ["Smoke detector, smoke alarm", "Siren"])
def test_default_monitored_classes_cover_fire_alarms(class_name):
    from demo.safety import DEFAULT_MONITORED_CLASSES

    assert class_name in DEFAULT_MONITORED_CLASSES


def test_a_stale_alarm_reason_does_not_leak_into_the_next_session():
    """A release racing teardown must not label the next stop a fire alarm."""
    runner, _ = running_runner()
    runner._on_safety_change(True, SafetyEvent("Siren", 0.9))
    assert runner.close(timeout=5)

    # The alarm reason arrives after the session already finished.
    runner._stop_reason = "Stopped for safety: alarm detected (Siren)."

    runner.start()
    assert runner.close(timeout=5)

    assert "alarm" not in runner.state.snapshot().detail.lower()
