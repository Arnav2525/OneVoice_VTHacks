

from __future__ import annotations

import array
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo.safety import (
    FakeClassifier,
    SafetyEvent,
    SafetyGatedSink,
    SafetyMonitor,
    _chunk_to_float32,
)
from onevoice.core.models.audio_chunk import AudioChunk

def _chunk(
    n: int = 320, timestamp_ms: float = 0.0, sample_rate: int = 16_000
) -> AudioChunk:
    return AudioChunk(
        timestamp_ms=timestamp_ms,
        data=array.array("f", [0.0] * n),
        sample_rate=sample_rate,
        channels=1,
        metadata={},
    )

class StateCollector:

    def __init__(self) -> None:
        self.calls: list[tuple[bool, SafetyEvent | None]] = []
        self._event = threading.Event()

    def __call__(self, active: bool, event: SafetyEvent | None) -> None:
        self.calls.append((active, event))
        self._event.set()

    def wait(self, timeout: float = 2.0) -> bool:
        got = self._event.wait(timeout)
        self._event.clear()
        return got

def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()

def test_chunk_to_float32_converts_array_array() -> None:
    chunk = AudioChunk(
        timestamp_ms=0.0,
        data=array.array("f", [0.1, -0.2, 0.3]),
        sample_rate=16_000,
        channels=1,
        metadata={},
    )
    out = _chunk_to_float32(chunk)
    assert out.dtype.name == "float32"
    assert list(out) == [
        array.array("f", [0.1])[0],
        array.array("f", [-0.2])[0],
        array.array("f", [0.3])[0],
    ]

def test_chunk_to_float32_empty_chunk() -> None:
    chunk = _chunk(n=0)
    out = _chunk_to_float32(chunk)
    assert len(out) == 0

def test_event_below_threshold_does_nothing() -> None:
    classifier = FakeClassifier(script=[SafetyEvent("Siren", 0.2)])
    collector = StateCollector()
    monitor = SafetyMonitor(
        classifier,
        activate_thresh=0.5,
        window_s=0.02,
        hop_s=0.02,
        on_state_change=collector,
    )
    monitor.start()
    try:
        monitor.write(_chunk(timestamp_ms=0.0))
        assert classifier.wait_for_call(timeout=2.0)
        assert not _wait_until(lambda: monitor.is_active(), timeout=0.3)
        assert collector.calls == []
    finally:
        monitor.stop()

def test_event_above_threshold_activates_immediately() -> None:
    classifier = FakeClassifier(script=[SafetyEvent("Siren", 0.9)])
    collector = StateCollector()
    monitor = SafetyMonitor(
        classifier,
        activate_thresh=0.5,
        window_s=0.02,
        hop_s=0.02,
        on_state_change=collector,
    )
    monitor.start()
    try:
        monitor.write(_chunk(timestamp_ms=0.0))
        assert _wait_until(lambda: monitor.is_active())
        assert collector.calls == [(True, SafetyEvent("Siren", 0.9))]
    finally:
        monitor.stop()

def test_activation_persists_through_hold_then_releases_after_it_expires() -> None:
    classifier = FakeClassifier(
        script=[SafetyEvent("Siren", 0.9), None, None, None]
    )
    collector = StateCollector()
    monitor = SafetyMonitor(
        classifier,
        activate_thresh=0.5,
        release_hold_s=1.0,
        window_s=0.02,
        hop_s=0.02,
        on_state_change=collector,
    )
    monitor.start()
    try:
        monitor.write(_chunk(timestamp_ms=0.0))
        assert _wait_until(lambda: monitor.is_active())

        monitor.write(_chunk(timestamp_ms=500.0))
        time.sleep(0.05)
        assert monitor.is_active() is True

        monitor.write(_chunk(timestamp_ms=1600.0))
        assert _wait_until(lambda: not monitor.is_active())
        assert collector.calls[0] == (True, SafetyEvent("Siren", 0.9))
        assert collector.calls[-1] == (False, None)

        assert len(collector.calls) == 2
    finally:
        monitor.stop()

def test_monitored_classes_filters_out_unlisted_classes() -> None:
    classifier = FakeClassifier(
        script=[SafetyEvent("Dog bark", 0.95), SafetyEvent("Siren", 0.95)]
    )
    monitor = SafetyMonitor(
        classifier,
        monitored_classes={"Siren"},
        activate_thresh=0.5,
        window_s=0.02,
        hop_s=0.02,
    )
    monitor.start()
    try:
        monitor.write(_chunk(timestamp_ms=0.0))
        assert classifier.wait_for_call(timeout=2.0)
        assert not _wait_until(lambda: monitor.is_active(), timeout=0.3)

        classifier.reset()
        monitor.write(_chunk(timestamp_ms=100.0))
        assert _wait_until(lambda: monitor.is_active())
    finally:
        monitor.stop()

def test_latest_raw_chunk_returns_most_recent_regardless_of_active_state() -> None:
    classifier = FakeClassifier()
    monitor = SafetyMonitor(classifier, window_s=0.02, hop_s=0.02)
    monitor.start()
    try:
        chunk_a = _chunk(timestamp_ms=0.0)
        monitor.write(chunk_a)
        assert monitor.latest_raw_chunk() is chunk_a

        chunk_b = _chunk(timestamp_ms=20.0)
        monitor.write(chunk_b)
        assert monitor.latest_raw_chunk() is chunk_b
    finally:
        monitor.stop()

class _RaisingOnceClassifier:

    def __init__(self, then: FakeClassifier) -> None:
        self._invocation_count = 0
        self._then = then

    def classify(self, audio, sample_rate):
        self._invocation_count += 1
        if self._invocation_count == 1:
            raise RuntimeError("boom")
        return self._then.classify(audio, sample_rate)

def test_classifier_exception_does_not_kill_worker_thread() -> None:
    inner = FakeClassifier(script=[SafetyEvent("Siren", 0.9)])
    classifier = _RaisingOnceClassifier(inner)
    collector = StateCollector()
    monitor = SafetyMonitor(
        classifier,
        activate_thresh=0.5,
        window_s=0.02,
        hop_s=0.02,
        on_state_change=collector,
    )
    monitor.start()
    try:
        monitor.write(_chunk(timestamp_ms=0.0))
        assert _wait_until(lambda: classifier._invocation_count == 1)
        monitor.write(_chunk(timestamp_ms=20.0))
        assert _wait_until(lambda: monitor.is_active())
    finally:
        monitor.stop()

class _RaisingOnceCallback:

    def __init__(self) -> None:
        self._invocation_count = 0
        self.calls: list[tuple[bool, SafetyEvent | None]] = []
        self._event = threading.Event()

    def __call__(self, active: bool, event: SafetyEvent | None) -> None:
        self._invocation_count += 1
        if self._invocation_count == 1:
            raise RuntimeError("boom")
        self.calls.append((active, event))
        self._event.set()

    def wait(self, timeout: float = 2.0) -> bool:
        return self._event.wait(timeout)

def test_on_state_change_exception_does_not_kill_worker_thread() -> None:
    classifier = FakeClassifier(
        script=[SafetyEvent("Siren", 0.9), None, SafetyEvent("Siren", 0.9)]
    )
    callback = _RaisingOnceCallback()
    monitor = SafetyMonitor(
        classifier,
        activate_thresh=0.5,
        release_hold_s=0.01,
        window_s=0.02,
        hop_s=0.02,
        on_state_change=callback,
    )
    monitor.start()
    try:
        monitor.write(_chunk(timestamp_ms=0.0))
        assert _wait_until(lambda: callback._invocation_count == 1)
        monitor.write(_chunk(timestamp_ms=100.0))
        assert _wait_until(lambda: len(classifier.calls) >= 2)
        monitor.write(_chunk(timestamp_ms=200.0))
        assert callback.wait(timeout=2.0)
        assert callback.calls
    finally:
        monitor.stop()

class _RecordingSink:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.written: list[AudioChunk] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def write(self, chunk: AudioChunk) -> None:
        self.written.append(chunk)

class _FakeMonitor:

    def __init__(self) -> None:
        self.active = False
        self.raw_chunk: AudioChunk | None = None

    def is_active(self) -> bool:
        return self.active

    def latest_raw_chunk(self) -> AudioChunk | None:
        return self.raw_chunk

def test_sink_passes_through_isolated_chunk_when_inactive() -> None:
    real_sink = _RecordingSink()
    monitor = _FakeMonitor()
    monitor.active = False
    sink = SafetyGatedSink(real_sink, monitor)

    isolated = _chunk(timestamp_ms=0.0)
    sink.write(isolated)
    assert real_sink.written == [isolated]

def test_sink_substitutes_raw_chunk_when_active() -> None:
    real_sink = _RecordingSink()
    monitor = _FakeMonitor()
    monitor.active = True
    raw = _chunk(timestamp_ms=0.0)
    monitor.raw_chunk = raw
    sink = SafetyGatedSink(real_sink, monitor)

    isolated = _chunk(timestamp_ms=20.0)
    sink.write(isolated)
    assert real_sink.written == [raw]
    assert real_sink.written[0] is not isolated

def test_sink_falls_back_to_original_chunk_when_active_but_no_raw_chunk_yet() -> None:

    real_sink = _RecordingSink()
    monitor = _FakeMonitor()
    monitor.active = True
    monitor.raw_chunk = None
    sink = SafetyGatedSink(real_sink, monitor)

    isolated = _chunk(timestamp_ms=0.0)
    sink.write(isolated)
    assert real_sink.written == [isolated]

def test_sink_start_stop_delegate_to_real_sink() -> None:
    real_sink = _RecordingSink()
    sink = SafetyGatedSink(real_sink, _FakeMonitor())
    sink.start()
    sink.stop()
    assert real_sink.started is True
    assert real_sink.stopped is True
