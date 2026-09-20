import threading
import time

from onevoice.audio.io import MockAudioSink, MockAudioSource
from onevoice.selection.first_track_selector import FirstTrackSelector
from onevoice.separation.passthrough import PassthroughSeparator
from onevoice.streaming.pipeline import StreamingPipeline
from onevoice.video.capture import MockVideoSource
from onevoice.video.trackers.stub_tracker import StubFaceTracker


class FlakyVideoSource:
    def __init__(self, fail_at, fail_count):
        self._inner = MockVideoSource()
        self._fail_at = fail_at
        self._fail_count = fail_count
        self._reads = 0
        self._lock = threading.Lock()
        self.starts = 0
        self.stops = 0

    def start(self):
        with self._lock:
            self.starts += 1
        self._inner.start()

    def stop(self):
        with self._lock:
            self.stops += 1
        self._inner.stop()

    def read(self):
        with self._lock:
            self._reads += 1
            reads = self._reads
        if self._fail_at <= reads < self._fail_at + self._fail_count:
            raise RuntimeError("Webcam frame read failed")
        return self._inner.read()

    @property
    def reads(self):
        with self._lock:
            return self._reads


def _build(video_source):
    return StreamingPipeline(
        audio_source=MockAudioSource(),
        audio_sink=MockAudioSink(),
        video_source=video_source,
        face_tracker=StubFaceTracker(),
        target_selector=FirstTrackSelector(),
        target_separator=PassthroughSeparator(),
    )


def test_transient_read_failure_does_not_kill_video_capture():
    source = FlakyVideoSource(fail_at=3, fail_count=2)
    pipeline = _build(source)
    pipeline.start()
    time.sleep(1.2)
    reads_before_stop = source.reads
    pipeline.stop()

    stats = pipeline.get_stats()
    assert stats["video_read_failures"] == 2
    assert stats["video_reopens"] >= 1
    assert reads_before_stop > 5
    assert stats["results_processed"] > 0
    assert stats["threads_alive"] == 0


def test_persistent_read_failure_eventually_gives_up():
    source = FlakyVideoSource(fail_at=1, fail_count=10_000)
    pipeline = _build(source)
    pipeline._video_max_read_failures = 3
    pipeline._video_retry_delay_s = 0.0
    pipeline.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if pipeline.get_stats()["video_read_failures"] > 3:
            break
        time.sleep(0.05)
    pipeline.stop()

    stats = pipeline.get_stats()
    assert stats["video_read_failures"] == 4
    assert stats["threads_alive"] == 0


def test_reopen_returns_false_without_lifecycle_methods():
    class Bare:
        def read(self):
            return MockVideoSource().read()

    pipeline = _build(Bare())
    assert pipeline._reopen_video_source() is False
