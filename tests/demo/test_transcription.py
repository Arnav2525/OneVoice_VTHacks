

from __future__ import annotations

import array
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo.transcription import FakeTranscriber, TranscribingSink, _chunk_to_float32
from onevoice.core.models.audio_chunk import AudioChunk

def _chunk(
    samples: list[float], timestamp_ms: float = 0.0, sample_rate: int = 16_000
) -> AudioChunk:
    return AudioChunk(
        timestamp_ms=timestamp_ms,
        data=array.array("f", samples),
        sample_rate=sample_rate,
        channels=1,
        metadata={},
    )

class Collector:

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self._event = threading.Event()

    def __call__(self, label: str, text: str, timestamp_ms: float) -> None:
        self.calls.append((label, text, timestamp_ms))
        self._event.set()

    def wait(self, timeout: float = 2.0) -> bool:
        got = self._event.wait(timeout)
        self._event.clear()
        return got

def test_chunk_to_float32_converts_array_array() -> None:
    chunk = _chunk([0.1, -0.2, 0.3])
    result = _chunk_to_float32(chunk)
    assert result.dtype.name == "float32"
    assert list(result) == [
        0.10000000149011612,
        -0.20000000298023224,
        0.30000001192092896,
    ]

def test_chunk_to_float32_empty_chunk() -> None:
    chunk = _chunk([])
    result = _chunk_to_float32(chunk)
    assert len(result) == 0

def test_no_transcription_below_threshold() -> None:
    transcriber = FakeTranscriber()
    collector = Collector()
    sink = TranscribingSink(
        "test",
        transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=1.0,
        silence_thresh=None,
    )
    sink.start()
    try:

        sink.write(_chunk([0.05] * 100))
        assert not transcriber.wait_for_call(timeout=0.3)
        assert transcriber.calls == []
    finally:
        sink.stop()

def test_transcription_fires_once_threshold_reached() -> None:
    transcriber = FakeTranscriber(text="hello world")
    collector = Collector()
    sink = TranscribingSink(
        "isolated",
        transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=1.0,
        silence_thresh=None,
    )
    sink.start()
    try:
        sink.write(_chunk([0.1] * 16_000))
        assert transcriber.wait_for_call(timeout=2.0)
        assert collector.wait(timeout=2.0)
    finally:
        sink.stop()

    assert len(transcriber.calls) == 1
    num_samples, sample_rate = transcriber.calls[0]
    assert num_samples == 16_000
    assert sample_rate == 16_000
    assert collector.calls == [("isolated", "hello world", 0.0)]

def test_buffer_resets_after_flush_does_not_double_count() -> None:
    transcriber = FakeTranscriber()
    collector = Collector()
    sink = TranscribingSink(
        "raw",
        transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=1.0,
        silence_thresh=None,
    )
    sink.start()
    try:
        sink.write(_chunk([0.1] * 16_000))
        assert transcriber.wait_for_call(timeout=2.0)
        transcriber.reset()

        sink.write(_chunk([0.2] * 8_000))
        assert not transcriber.wait_for_call(timeout=0.3)

        sink.write(_chunk([0.2] * 8_000))
        assert transcriber.wait_for_call(timeout=2.0)
    finally:
        sink.stop()

    assert len(transcriber.calls) == 2

    assert transcriber.calls[1] == (16_000, 16_000)

def test_stop_flushes_remaining_partial_buffer() -> None:
    transcriber = FakeTranscriber(text="partial")
    collector = Collector()
    sink = TranscribingSink(
        "raw",
        transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=10.0,
        silence_thresh=None,
    )
    sink.start()
    sink.write(_chunk([0.3] * 500))
    sink.stop()

    assert len(transcriber.calls) == 1
    assert transcriber.calls[0][0] == 500
    assert collector.calls == [("raw", "partial", 0.0)]

def test_stop_with_empty_buffer_does_not_transcribe() -> None:
    transcriber = FakeTranscriber()
    collector = Collector()
    sink = TranscribingSink("raw", transcriber, collector, sample_rate=16_000)
    sink.start()
    sink.stop()

    assert transcriber.calls == []
    assert collector.calls == []

def test_blank_transcript_is_not_forwarded_to_on_transcript() -> None:

    transcriber = FakeTranscriber(text="   ")
    collector = Collector()
    sink = TranscribingSink(
        "raw",
        transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=10.0,
        silence_hold_s=0.1,
        min_segment_s=0.05,
    )
    sink.start()
    sink.write(_chunk([0.5] * 1_600))
    sink.write(_chunk([0.0] * 1_600))
    assert transcriber.wait_for_call(timeout=2.0)
    sink.stop()

    assert len(transcriber.calls) == 1
    assert collector.calls == []

def test_transcriber_exception_does_not_kill_worker_thread() -> None:
    class FlakyTranscriber:
        def __init__(self) -> None:
            self.call_count = 0

        def transcribe(self, audio, sample_rate):  # noqa: ANN001
            self.call_count += 1
            if self.call_count == 1:
                raise RuntimeError("boom")
            return "recovered"

    transcriber = FlakyTranscriber()
    collector = Collector()
    sink = TranscribingSink(
        "raw",
        transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=1.0,
        silence_thresh=None,
    )
    sink.start()
    try:
        sink.write(_chunk([0.1] * 16_000))
        assert collector.wait(timeout=2.0)
        sink.write(_chunk([0.1] * 16_000))
        assert collector.wait(timeout=2.0)
    finally:
        sink.stop()

    assert transcriber.call_count == 2
    assert collector.calls[0][1].startswith("[transcription error:")
    assert collector.calls[1] == ("raw", "recovered", 0.0)

def test_on_transcript_exception_does_not_kill_worker_thread() -> None:

    class FlakyCallback:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, float]] = []
            self._invocation_count = 0

        def __call__(self, label: str, text: str, timestamp_ms: float) -> None:
            self._invocation_count += 1
            if self._invocation_count == 1:
                raise UnicodeEncodeError(
                    "cp1252", text, 0, 1, "character maps to <undefined>"
                )
            self.calls.append((label, text, timestamp_ms))

    on_transcript = FlakyCallback()
    transcriber = FakeTranscriber(text="first segment")
    sink = TranscribingSink(
        "isolated",
        transcriber,
        on_transcript,
        sample_rate=16_000,
        max_segment_s=1.0,
        silence_thresh=None,
    )
    sink.start()
    try:
        sink.write(
            _chunk([0.1] * 16_000)
        )
        assert transcriber.wait_for_call(timeout=2.0)
        transcriber.reset()
        transcriber.text = "second segment"
        sink.write(_chunk([0.1] * 16_000))
        assert transcriber.wait_for_call(timeout=2.0)
    finally:
        sink.stop()

    assert transcriber.calls == [
        (16_000, 16_000),
        (16_000, 16_000),
    ]
    assert on_transcript.calls == [
        ("isolated", "second segment", 0.0)
    ]

def test_two_independent_sinks_do_not_interfere() -> None:

    raw_transcriber = FakeTranscriber(text="raw text")
    isolated_transcriber = FakeTranscriber(text="isolated text")
    collector = Collector()

    raw_sink = TranscribingSink(
        "raw",
        raw_transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=1.0,
        silence_thresh=None,
    )
    isolated_sink = TranscribingSink(
        "isolated",
        isolated_transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=1.0,
        silence_thresh=None,
    )
    raw_sink.start()
    isolated_sink.start()
    try:
        raw_sink.write(_chunk([0.1] * 16_000))
        isolated_sink.write(_chunk([0.2] * 16_000))
        assert raw_transcriber.wait_for_call(timeout=2.0)
        assert isolated_transcriber.wait_for_call(timeout=2.0)
    finally:
        raw_sink.stop()
        isolated_sink.stop()

    labels = {label for label, _, _ in collector.calls}
    assert labels == {"raw", "isolated"}

def test_silence_gap_flushes_before_ceiling_reached() -> None:

    transcriber = FakeTranscriber(text="hi there")
    collector = Collector()
    sink = TranscribingSink(
        "raw",
        transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=10.0,
        silence_hold_s=0.2,
        min_segment_s=0.3,
    )
    sink.start()
    try:
        sink.write(_chunk([0.5] * 6_400))
        assert not transcriber.wait_for_call(timeout=0.3)
        sink.write(_chunk([0.0] * 4_000))
        assert transcriber.wait_for_call(timeout=2.0)
    finally:
        sink.stop()

    assert len(transcriber.calls) == 1
    num_samples, _ = transcriber.calls[0]
    assert num_samples == 6_400 + 4_000

def test_silence_gap_does_not_fire_below_min_segment_s() -> None:

    transcriber = FakeTranscriber()
    collector = Collector()
    sink = TranscribingSink(
        "raw",
        transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=10.0,
        silence_hold_s=0.1,
        min_segment_s=1.0,
    )
    sink.start()
    try:
        sink.write(_chunk([0.5] * 800))
        sink.write(_chunk([0.0] * 3_200))

        assert not transcriber.wait_for_call(timeout=0.3)
    finally:
        sink.stop()

    assert len(transcriber.calls) == 1
    assert transcriber.calls[0][0] == 800 + 3_200

def test_leading_silence_is_dropped_without_buffering() -> None:

    transcriber = FakeTranscriber()
    collector = Collector()
    sink = TranscribingSink(
        "raw",
        transcriber,
        collector,
        sample_rate=16_000,
        max_segment_s=0.1,
    )
    sink.start()
    try:
        sink.write(_chunk([0.0] * 32_000))
        assert not transcriber.wait_for_call(timeout=0.3)
    finally:
        sink.stop()

    assert transcriber.calls == []
    assert collector.calls == []
