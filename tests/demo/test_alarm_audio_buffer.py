import numpy as np

from demo.safety import FakeClassifier, SafetyMonitor
from onevoice.core.models.audio_chunk import AudioChunk


def chunk(values, timestamp=1000):
    return AudioChunk(timestamp, list(values), 16000, 1, {})


def test_alarm_playback_preserves_order_across_different_chunk_sizes():
    monitor = SafetyMonitor(FakeClassifier())
    monitor._active = True
    monitor.write(chunk(np.arange(320)))
    monitor.write(chunk(np.arange(320, 640), 1020))
    first = monitor.take_playback(chunk(np.zeros(480)))
    second = monitor.take_playback(chunk(np.zeros(160)))
    np.testing.assert_array_equal(first.data + second.data, np.arange(640))
    empty = monitor.take_playback(chunk(np.zeros(320)))
    assert len(empty.data) == 320
    assert not any(empty.data)
    assert empty.metadata["alarm_underrun_samples"] == 320


def test_slow_classifier_retains_only_the_newest_pending_window():
    monitor = SafetyMonitor(FakeClassifier(), window_s=0.02, hop_s=0.02)
    for i in range(20):
        monitor.write(chunk([float(i)] * 320, i * 20))
    assert monitor._queue.qsize() == 1
    samples, timestamp = monitor._queue.get_nowait()
    assert timestamp == 380
    assert np.all(samples == 19)
    monitor.stop()
