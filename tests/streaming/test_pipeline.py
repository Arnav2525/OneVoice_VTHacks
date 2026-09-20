

import threading
import time

import pytest

from onevoice.audio.io import MockAudioSink, MockAudioSource
from onevoice.selection.first_track_selector import FirstTrackSelector
from onevoice.separation.passthrough import PassthroughSeparator
from onevoice.separation.registry import create_separator
from onevoice.streaming.pipeline import StreamingPipeline
from onevoice.video.capture import MockVideoSource
from onevoice.video.trackers.stub_tracker import StubFaceTracker

def _build(separator):
    return StreamingPipeline(
        audio_source=MockAudioSource(),
        audio_sink=MockAudioSink(),
        video_source=MockVideoSource(),
        face_tracker=StubFaceTracker(),
        target_selector=FirstTrackSelector(),
        target_separator=separator,
    )

def test_target_separator_property_exposes_backend():
    separator = PassthroughSeparator()
    pipeline = _build(separator)
    assert pipeline.target_separator is separator

def test_lifecycle_and_results():
    pipeline = _build(PassthroughSeparator())
    pipeline.start()
    assert pipeline.is_running
    time.sleep(0.6)
    pipeline.stop()
    assert not pipeline.is_running

    stats = pipeline.get_stats()
    assert stats["results_processed"] > 0
    assert stats["threads_total"] == 6
    assert stats["audio_drops"] >= 0

    assert stats["threads_alive"] == 0

def test_double_start_is_idempotent():
    pipeline = _build(PassthroughSeparator())
    pipeline.start()
    pipeline.start()
    try:
        assert stats_thread_count(pipeline) == 6
    finally:
        pipeline.stop()

    pipeline.stop()
    assert not pipeline.is_running

def stats_thread_count(pipeline):
    return pipeline.get_stats()["threads_total"]

def test_get_backend_status_reports_passthrough():
    pipeline = _build(PassthroughSeparator())
    status = pipeline.get_backend_status()
    assert status["backend"] == "passthrough"
    assert status["is_real_separation"] is False

def test_start_warns_on_passthrough(caplog):
    pipeline = _build(PassthroughSeparator())
    with caplog.at_level("WARNING"):
        pipeline.start()
        pipeline.stop()
    assert any("NO REAL SEPARATION" in r.getMessage() for r in caplog.records)
    stats = pipeline.get_stats()
    assert stats["separator_backend"] == "passthrough"
    assert stats["separator_is_real_separation"] is False

def test_pipeline_surfaces_backend_fallback():

    separator = create_separator({"name": "asteroid", "device": "cpu"})
    pipeline = _build(separator)
    pipeline.start()
    time.sleep(0.5)
    pipeline.stop()

    stats = pipeline.get_stats()
    assert stats["separator_backend"] == "asteroid"
    assert stats["separator_fallback_active"] is True
    assert stats["separator_healthy"] is False

class _BlockingDevice:

    def __init__(self, name, *, fail_start=False, fail_stop=False):
        self.name = name
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.started = 0
        self.stopped = 0
        self.open = False
        self.reading = threading.Event()
        self.released = threading.Event()
        self.start_error = RuntimeError(f"{name} failed to start")

    def start(self):
        self.started += 1
        self.open = True
        if self.fail_start:
            raise self.start_error

    def stop(self):
        self.stopped += 1
        self.open = False
        self.released.set()
        if self.fail_stop:
            self.fail_stop = False
            raise RuntimeError(f"{self.name} failed to stop")

    def read(self):
        self.reading.set()
        self.released.wait()
        raise RuntimeError("Device closed")

    def write(self, chunk):
        pass

def _build_devices(devices):
    return StreamingPipeline(
        audio_source=devices[0],
        audio_sink=devices[1],
        video_source=devices[2],
        face_tracker=StubFaceTracker(),
        target_selector=FirstTrackSelector(),
        target_separator=PassthroughSeparator(),
    )

@pytest.mark.parametrize("failure_index", [0, 1, 2])
def test_start_failure_closes_every_attempted_device(failure_index):
    devices = [_BlockingDevice(name) for name in ("mic", "speaker", "camera")]
    failed_device = devices[failure_index]
    failed_device.fail_start = True
    pipeline = _build_devices(devices)

    with pytest.raises(RuntimeError) as error:
        pipeline.start()
    assert error.value is failed_device.start_error
    assert not pipeline.is_running
    assert all(not device.open for device in devices)
    assert [device.stopped for device in devices] == [
        int(index <= failure_index) for index in range(3)
    ]
    pipeline.stop()

def test_stop_before_start_and_repeated_stop_do_not_touch_unopened_devices():
    devices = [_BlockingDevice(name) for name in ("mic", "speaker", "camera")]
    pipeline = _build_devices(devices)
    pipeline.stop()
    pipeline.stop()
    assert [device.stopped for device in devices] == [0, 0, 0]
    assert not pipeline.is_running

def test_stop_unblocks_capture_and_does_not_hold_state_lock():
    devices = [_BlockingDevice(name) for name in ("mic", "speaker", "camera")]
    pipeline = _build_devices(devices)
    original_read = devices[0].read

    def read_then_inspect_state():
        try:
            original_read()
        finally:

            with pipeline._lock:
                pass

    devices[0].read = read_then_inspect_state
    pipeline.start()
    try:
        assert devices[0].reading.wait(1.0)
        assert devices[2].reading.wait(1.0)
    finally:
        pipeline.stop()
    assert pipeline.get_stats()["threads_alive"] == 0
    assert [device.stopped for device in devices] == [1, 1, 1]

def test_one_stop_failure_still_releases_other_devices_and_can_be_retried():
    devices = [_BlockingDevice(name) for name in ("mic", "speaker", "camera")]
    devices[2].fail_stop = True
    pipeline = _build_devices(devices)
    pipeline.start()
    with pytest.raises(RuntimeError, match="camera failed to stop"):
        pipeline.stop()
    assert [device.stopped for device in devices] == [1, 1, 1]
    assert pipeline.get_stats()["threads_alive"] == 0
    assert pipeline.is_running
    pipeline.stop()
    assert not pipeline.is_running
    assert [device.stopped for device in devices] == [1, 1, 2]

def test_start_error_survives_rollback_error_and_cleanup_can_be_retried():
    devices = [_BlockingDevice(name) for name in ("mic", "speaker", "camera")]
    devices[0].fail_stop = True
    devices[2].fail_start = True
    pipeline = _build_devices(devices)
    with pytest.raises(RuntimeError) as error:
        pipeline.start()
    assert error.value is devices[2].start_error
    assert [device.stopped for device in devices] == [1, 1, 1]
    pipeline.stop()
    assert not pipeline.is_running

def test_worker_that_does_not_stop_is_reported_and_blocks_restart():
    devices = [_BlockingDevice(name) for name in ("mic", "speaker", "camera")]
    release_worker = threading.Event()

    def stuck_read():
        devices[0].reading.set()
        release_worker.wait()
        raise RuntimeError("Released by test")

    devices[0].read = stuck_read
    pipeline = _build_devices(devices)
    pipeline._stop_timeout_s = 0.02
    pipeline.start()
    try:
        assert devices[0].reading.wait(1.0)
        with pytest.raises(RuntimeError, match="onevoice-audio-capture"):
            pipeline.stop()
        assert pipeline.get_stats()["threads_alive"] >= 1
        assert pipeline.is_running
        with pytest.raises(RuntimeError, match="cleanup is incomplete"):
            pipeline.start()
    finally:
        release_worker.set()
        pipeline._stop_timeout_s = 2.0
        pipeline.stop()
    assert pipeline.get_stats()["threads_alive"] == 0

def test_thread_start_failure_rolls_back_devices_and_started_workers(monkeypatch):
    devices = [_BlockingDevice(name) for name in ("mic", "speaker", "camera")]
    pipeline = _build_devices(devices)
    original_start = threading.Thread.start

    def start_thread(thread):
        if thread.name == "onevoice-video-capture":
            raise RuntimeError("Cannot create video thread")
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", start_thread)
    with pytest.raises(RuntimeError, match="Cannot create video thread"):
        pipeline.start()
    assert [device.stopped for device in devices] == [1, 1, 1]
    assert not pipeline.is_running
    assert pipeline.get_stats()["threads_alive"] == 0
