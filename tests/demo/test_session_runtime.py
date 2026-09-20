

from __future__ import annotations

import array
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from demo import session_runtime as runtime  # noqa: E402
from demo.session_state import SessionState  # noqa: E402
from onevoice.core.models.audio_chunk import AudioChunk  # noqa: E402
from onevoice.core.models.frame import Frame  # noqa: E402
from onevoice.core.models.speaker_track import SpeakerTrack  # noqa: E402
from onevoice.core.models.target_selection import TargetSelection  # noqa: E402

def audio(**metadata):
    return AudioChunk(10000.0, array.array("f", [0.25] * 32), 16000, 1, metadata)

def track(name):
    return SpeakerTrack(name, (0, 0, 100, 100), 1.0, {"visible": True})

class FakePipeline:
    def __init__(self, stop_release=None):
        self.started = threading.Event()
        self.stop_entered = threading.Event()
        self.stopped = threading.Event()
        self.stop_release = stop_release

    def start(self):
        self.started.set()

    def stop(self):
        self.stop_entered.set()
        if self.stop_release is not None:
            assert self.stop_release.wait(timeout=5), "Test cleanup was not released"
        self.stopped.set()

class FakeSeparator:
    def __init__(self, load_release=None):
        self.loading = threading.Event()
        self.closed = threading.Event()
        self.load_release = load_release

    def load(self):
        self.loading.set()
        if self.load_release is not None:
            assert self.load_release.wait(timeout=5), "Test model load was not released"

    def close(self):
        self.closed.set()

def test_ready_opens_nothing_and_copies_config(tmp_path):
    factory = Mock(side_effect=AssertionError("Must wait for Start listening"))
    config = {"backend": {"name": "dolphin"}}
    runner = runtime.SessionRunner(config, factory=factory, record_root=tmp_path)
    config["backend"]["name"] = "modified"
    assert runner.config["backend"]["name"] == "dolphin"
    assert runner.state.snapshot().phase == "ready"
    assert runner.frame() is None
    assert not runner.busy
    assert not list(tmp_path.iterdir())
    assert runner.close(timeout=1)
    factory.assert_not_called()

@pytest.mark.parametrize("blocked_stage", ["factory", "model_load"])
def test_start_is_async_and_cancel_prevents_late_device_start(blocked_stage):
    entered, release = threading.Event(), threading.Event()
    pipeline = FakePipeline()
    separator = FakeSeparator(release if blocked_stage == "model_load" else None)
    factory_threads = []

    def factory():
        factory_threads.append(threading.get_ident())
        entered.set()
        if blocked_stage == "factory":
            assert release.wait(timeout=5)
        return pipeline, separator

    runner = runtime.SessionRunner({}, factory=factory)
    try:
        runner.start()
        assert entered.wait(timeout=5)
        if blocked_stage == "model_load":
            assert separator.loading.wait(timeout=5)
        assert factory_threads == [runner._thread.ident]
        assert factory_threads[0] != threading.get_ident()
        assert runner.state.snapshot().phase == "starting"
        runner.stop()
        assert runner.state.snapshot().phase == "stopping"
        assert not runner.close(timeout=0.01)
        assert not pipeline.started.is_set()
    finally:
        release.set()
        assert runner.close(timeout=5)
    assert not pipeline.started.is_set()
    assert pipeline.stopped.is_set()
    assert separator.closed.is_set()
    assert runner.state.snapshot().phase == "stopped"

def test_stopping_remains_pending_until_device_cleanup_finishes():
    release = threading.Event()
    pipeline, separator = FakePipeline(release), FakeSeparator()
    runner = runtime.SessionRunner({}, factory=lambda: (pipeline, separator))
    try:
        runner.start()
        assert pipeline.started.wait(timeout=5)
        runner.stop()
        assert pipeline.stop_entered.wait(timeout=5)
        assert runner.busy
        assert runner.state.snapshot().phase == "stopping"
        assert not runner.close(timeout=0.01)
        assert not separator.closed.is_set()
    finally:
        release.set()
        assert runner.close(timeout=5)
    assert pipeline.stopped.is_set()
    assert separator.closed.is_set()
    assert runner.state.snapshot().phase == "stopped"

class RetryPipeline(FakePipeline):
    def __init__(self):
        super().__init__()
        self.allow_stop = False
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        self.stop_entered.set()
        if not self.allow_stop:
            raise RuntimeError("Pipeline worker is still running")
        super().stop()

class RetrySeparator(FakeSeparator):
    def __init__(self):
        super().__init__()
        self.allow_close = False
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if not self.allow_close:
            raise RuntimeError("Model cleanup failed")
        super().close()

def join_session_worker(runner):
    runner._thread.join(timeout=5)
    assert not runner.busy

def test_failed_pipeline_stop_retains_resources_and_start_retries_before_rebuild():
    old_pipeline, old_separator = RetryPipeline(), FakeSeparator()
    new_pipeline, new_separator = FakePipeline(), FakeSeparator()
    factory = Mock(
        side_effect=[
            (old_pipeline, old_separator),
            (new_pipeline, new_separator),
        ]
    )
    runner = runtime.SessionRunner({}, factory=factory)
    try:
        runner.start()
        assert old_pipeline.started.wait(timeout=5)
        runner.stop()
        join_session_worker(runner)
        assert runner.state.snapshot().phase == "error"
        assert "still running" in runner.state.snapshot().detail
        assert runner._pipeline is old_pipeline
        assert runner._separator is old_separator
        assert not old_separator.closed.is_set()
        assert not runner.close(timeout=5)

        runner.start()
        join_session_worker(runner)
        assert factory.call_count == 1
        assert not old_separator.closed.is_set()

        old_pipeline.allow_stop = True
        runner.start()
        assert new_pipeline.started.wait(timeout=5)
        assert old_pipeline.stopped.is_set()
        assert old_separator.closed.is_set()
        assert factory.call_count == 2
    finally:
        old_pipeline.allow_stop = True
        assert runner.close(timeout=5)

def test_failed_separator_close_is_retained_without_stopping_pipeline_twice():
    pipeline, separator = FakePipeline(), RetrySeparator()
    pipeline.stop = Mock(wraps=pipeline.stop)
    factory = Mock(return_value=(pipeline, separator))
    runner = runtime.SessionRunner({}, factory=factory)
    try:
        runner.start()
        assert pipeline.started.wait(timeout=5)
        runner.stop()
        join_session_worker(runner)
        assert runner._pipeline is None
        assert runner._separator is separator
        assert pipeline.stop.call_count == 1
        assert not runner.close(timeout=5)
        runner.start()
        join_session_worker(runner)
        assert factory.call_count == 1
        assert pipeline.stop.call_count == 1
        assert runner.state.snapshot().phase == "error"

        separator.allow_close = True
        runner.stop()
        join_session_worker(runner)
        assert runner.state.snapshot().phase == "stopped"
        assert separator.closed.is_set()
        assert factory.call_count == 1
        assert runner.close(timeout=5)
    finally:
        separator.allow_close = True
        assert runner.close(timeout=5)

def test_cancel_during_cleanup_retry_does_not_start_a_new_pipeline():
    pipeline, separator = RetryPipeline(), FakeSeparator()
    factory = Mock(return_value=(pipeline, separator))
    runner = runtime.SessionRunner({}, factory=factory)
    release_cleanup = threading.Event()
    try:
        runner.start()
        assert pipeline.started.wait(timeout=5)
        runner.stop()
        join_session_worker(runner)

        pipeline.allow_stop = True
        pipeline.stop_release = release_cleanup
        pipeline.stop_entered.clear()
        runner.start()
        assert pipeline.stop_entered.wait(timeout=5)
        assert runner.state.snapshot().phase == "stopping"
        runner.stop()
        release_cleanup.set()
        join_session_worker(runner)
        assert factory.call_count == 1
        assert separator.closed.is_set()
        assert runner.state.snapshot().phase == "stopped"
    finally:
        pipeline.allow_stop = True
        release_cleanup.set()
        assert runner.close(timeout=5)

def test_failed_start_and_failed_cleanup_keep_original_error_and_resources():
    pipeline, separator = RetryPipeline(), FakeSeparator()
    pipeline.start = Mock(side_effect=RuntimeError("Camera failed to open"))
    runner = runtime.SessionRunner({}, factory=lambda: (pipeline, separator))
    try:
        runner.start()
        join_session_worker(runner)
        detail = runner.state.snapshot().detail
        assert "Camera failed to open" in detail
        assert "Pipeline worker is still running" in detail
        assert runner._pipeline is pipeline
        assert runner._separator is separator
        assert not separator.closed.is_set()
    finally:
        pipeline.allow_stop = True
        assert runner.close(timeout=5)
    assert separator.closed.is_set()

def test_restart_builds_fresh_pipeline_and_clears_target_and_frames(monkeypatch):
    built = []
    running = threading.Event()
    runner = runtime.SessionRunner({})
    mark_running = runner.state.mark_running

    def mark_and_notify():
        mark_running()
        running.set()

    def factory():
        pair = FakePipeline(), FakeSeparator()
        built.append(pair)
        return pair

    monkeypatch.setattr(runner.state, "mark_running", mark_and_notify)
    runner._factory = factory
    try:
        runner.start()
        assert running.wait(timeout=5)
        runner._on_frame(Frame(1000, None, {}))
        runner.state.observe_tracks([track("old")])
        assert runner.state.select("old")
        first_epoch = runner.state.selection()[1]
        assert runner.frame() is not None
        assert runner.close(timeout=5)
        running.clear()
        runner.start()
        assert running.wait(timeout=5)
        assert len(built) == 2
        assert built[0][0] is not built[1][0]
        assert built[0][1].closed.is_set()
        assert runner.frame() is None
        assert runner.selector.latest_tracks() == []
        assert runner.state.selection()[0] is None
        assert runner.state.selection()[1] > first_epoch
        runner.state.observe_tracks([track("new")])
        assert runner.state.person_name("new") == "Person 1"
    finally:
        assert runner.close(timeout=5)

def test_face_profiles_are_closed_only_after_pipeline_workers_stop():
    runner = runtime.SessionRunner({})
    pipeline = RetryPipeline()
    tracker = SimpleNamespace(close=Mock(), status=lambda: {"registered": 3})
    runner._pipeline = pipeline
    runner._identity_tracker = tracker
    assert runner.identity_status() == {"registered": 3}
    assert runner._cleanup_resources() is not None
    tracker.close.assert_not_called()
    assert runner._identity_tracker is tracker
    pipeline.allow_stop = True
    assert runner._cleanup_resources() is None
    tracker.close.assert_called_once()
    assert runner.identity_status() == {}

def test_live_builder_installs_local_identity_matching_before_devices_start(
    monkeypatch,
):
    import demo.face_identity as identity

    raw_tracker = SimpleNamespace(close=Mock())
    encoder = Mock()
    monkeypatch.setattr(identity, "LocalFaceEncoder", Mock(return_value=encoder))
    monkeypatch.setattr(runtime, "build_face_tracker", Mock(return_value=raw_tracker))
    runner = runtime.SessionRunner({}, live=True, preview=True)
    pipeline, separator = runner._build()
    runner._pipeline, runner._separator = pipeline, separator
    assert isinstance(pipeline._face_tracker, identity.SessionFaceTracker)
    assert pipeline._face_tracker.encoder is encoder
    assert pipeline._face_tracker.tracker is raw_tracker
    assert not pipeline.is_running
    assert runner._cleanup_resources() is None
    raw_tracker.close.assert_called_once()

def test_mock_pipeline_supports_third_target_without_gpu_or_isolation(monkeypatch):
    config = {"backend": {"name": "dolphin"}, "audio": {"chunk_samples": 320}}
    runner = runtime.SessionRunner(config)
    ready, third = threading.Event(), threading.Event()
    views, delivered, backend_configs = [], [], []
    original_observer = runner.state.observe_output
    original_builder = runtime.build_separator

    def observe(chunk, status):
        original_observer(chunk, status)
        view = runner.state.snapshot()
        views.append(view)
        if view.phase == "listening" and view.face_count == 3:
            ready.set()
            if view.selected_id == "preview-2":
                third.set()

    def build_cpu_only(settings):
        backend_configs.append(settings["backend"].copy())
        assert settings["backend"]["name"] == "passthrough"
        return original_builder(settings)

    monkeypatch.setattr(runner.state, "observe_output", observe)
    monkeypatch.setattr(runner.recorder, "on_output", delivered.append)
    monkeypatch.setattr(runtime, "build_separator", build_cpu_only)
    try:
        runner.start()
        assert ready.wait(timeout=5), runner.state.snapshot()
        assert runner.state.select("preview-2")
        assert third.wait(timeout=5), runner.state.snapshot()
        assert runner.frame().metadata["mock"]
        assert len(runner.selector.latest_tracks()) == 3
        assert runner.state.snapshot().person == "Person 3"
        assert runner.state.snapshot().mode == "preview"
        assert all(view.phase != "isolating" for view in views)
        assert all(view.output_level == 0 for view in views)
        assert delivered
        assert all(part.metadata["ui_muted"] for part in delivered)
        assert all(not any(part.data) for part in delivered)
        assert backend_configs == [{"name": "passthrough"}]
        assert runner.config == config
    finally:
        assert runner.close(timeout=5)

def prepared_sink(mode="live"):
    state = SessionState(mode, clock=lambda: 10.0)
    state.begin_start()
    state.mark_running()
    state.observe_input(audio())
    state.observe_frame(Frame(10000, [[[0, 0, 0]]], {}))
    state.observe_tracks([track("a"), track("b")])
    assert state.select("a")
    status = {"is_real_separation": True, "fallback_active": False}
    destination, recorded = [], []
    sink = runtime.SessionSink(
        SimpleNamespace(write=destination.append),
        state,
        SimpleNamespace(get_status=lambda: status),
        SimpleNamespace(on_output=recorded.append),
        SimpleNamespace(on_output=lambda chunk, allowed: None),
    )
    valid = audio(
        target_track_id="a",
        conditioning="visual",
        output_ready=True,
        ui_selection_epoch=state.selection()[1],
        fallback=False,
    )
    return sink, state, valid, status, destination, recorded

@pytest.mark.parametrize(
    "reason",
    [
        "wrong_target",
        "old_epoch",
        "fallback_chunk",
        "passthrough",
        "unconditioned",
        "not_ready",
        "fallback_backend",
        "unreal_backend",
        "preview",
    ],
)
def test_sink_mutes_ineligible_audio_even_if_backend_reports_real(reason):
    sink, state, chunk, status, destination, recorded = prepared_sink(
        "preview" if reason == "preview" else "live",
    )
    changes = {
        "wrong_target": {"target_track_id": "b"},
        "old_epoch": {"ui_selection_epoch": chunk.metadata["ui_selection_epoch"] - 1},
        "fallback_chunk": {"fallback": True},
        "passthrough": {"passthrough": True},
        "unconditioned": {"conditioning": "none"},
        "not_ready": {"output_ready": False},
    }
    chunk = replace(chunk, metadata={**chunk.metadata, **changes.get(reason, {})})
    if reason == "fallback_backend":
        status["fallback_active"] = True
    if reason == "unreal_backend":
        status["is_real_separation"] = False
    sink.write(chunk)
    assert len(destination) == 1
    assert not any(destination[0].data)
    assert destination[0].metadata["ui_muted"]
    assert destination[0].metadata["output_ready"] is False
    assert recorded == destination
    assert any(chunk.data)
    assert state.snapshot().phase != "isolating"
    assert state.snapshot().output_level == 0

def test_sink_delivers_current_valid_output_and_marks_hearing():
    sink, state, chunk, _, destination, recorded = prepared_sink()

    sink.write(chunk)
    assert destination[0].metadata["mute_ramp"] == "in"
    assert destination[0].data[-1] == pytest.approx(0.25, rel=0.05)

    sink.write(chunk)
    assert destination[1] == chunk
    assert "mute_ramp" not in destination[1].metadata
    assert recorded == destination
    assert state.snapshot().phase == "isolating"
    assert state.snapshot().title == "Hearing Person 1"
    assert state.snapshot().output_level == pytest.approx(0.25)

def test_cleanup_is_recorded_and_cannot_leak_after_muting():
    from onevoice.audio.cleanup import RumbleFilter

    sink, state, chunk, _, destination, recorded = prepared_sink()
    sink.cleanup = RumbleFilter()
    sink.write(chunk)
    assert destination[-1].metadata["rumble_filter_hz"] == 80.0
    assert destination[-1].data != chunk.data
    assert recorded == destination
    state.select(None)
    sink.write(chunk)

    assert destination[-1].metadata["ui_muted"]
    faded = list(destination[-1].data)
    assert abs(faded[-1]) < 1e-3
    assert all(abs(b) <= abs(a) for a, b in zip(faded, faded[1:]))
    sink.write(chunk)
    assert not any(destination[-1].data)
    state.select("a")
    silence = replace(
        chunk,
        data=[0.0] * len(chunk.data),
        metadata={
            **chunk.metadata,
            "ui_selection_epoch": state.selection()[1],
        },
    )
    sink.write(silence)
    assert not any(destination[-1].data)

def test_selection_change_during_cleanup_is_muted():
    sink, state, chunk, _, destination, _ = prepared_sink()

    def process(value):
        state.select("b")
        return value

    sink.cleanup = SimpleNamespace(process=process, reset=lambda: None)
    sink.write(chunk)
    assert not any(destination[-1].data)

def test_denoiser_runs_after_cleanup_and_records_processed_audio():
    sink, _, chunk, _, destination, recorded = prepared_sink()
    order = []

    def highpass(value):
        order.append("highpass")
        return replace(value, data=[0.1] * len(value.data))

    def denoise(value):
        order.append("denoise")
        assert value.data == [0.1] * len(value.data)
        return replace(
            value,
            data=[0.05] * len(value.data),
            metadata={
                **value.metadata,
                "noise_suppression": "gtcrn",
            },
        )

    sink.cleanup = SimpleNamespace(process=highpass, reset=Mock())
    sink.denoiser = SimpleNamespace(process=denoise, reset=Mock())
    sink.write(chunk)
    sink.write(chunk)
    assert order == ["highpass", "denoise", "highpass", "denoise"]
    assert recorded == destination
    assert destination[-1].data == [0.05] * len(chunk.data)
    assert destination[-1].metadata["noise_suppression"] == "gtcrn"

def test_selection_change_during_denoising_mutes_and_resets():
    sink, state, chunk, _, destination, recorded = prepared_sink()
    reset = Mock()

    def process(value):
        state.select("b")
        return value

    sink.denoiser = SimpleNamespace(process=process, reset=reset)
    sink.write(chunk)
    assert not any(destination[-1].data)
    assert recorded == destination
    reset.assert_called_once()

def test_muted_audio_never_enters_denoiser_and_resets_its_memory():
    sink, state, chunk, _, destination, _ = prepared_sink()
    sink.denoiser = SimpleNamespace(process=Mock(), reset=Mock())
    state.select(None)
    sink.write(chunk)
    sink.denoiser.process.assert_not_called()
    sink.denoiser.reset.assert_called_once()
    assert not any(destination[-1].data)

def test_denoiser_error_never_sends_unfiltered_audio_to_playback():
    sink, _, chunk, _, destination, recorded = prepared_sink()
    sink.denoiser = SimpleNamespace(
        process=Mock(side_effect=RuntimeError("Denoiser failed")),
        reset=Mock(),
    )
    with pytest.raises(RuntimeError, match="Denoiser failed"):
        sink.write(chunk)
    assert not destination
    assert not recorded

def test_live_build_uses_requested_camera_and_microphone(monkeypatch):
    from onevoice.audio import io
    from onevoice.video import capture

    microphone, speaker, camera = Mock(), Mock(), Mock()
    monkeypatch.setattr(io, "MicrophoneSource", microphone)
    monkeypatch.setattr(io, "SpeakerSink", speaker)
    monkeypatch.setattr(capture, "WebcamSource", camera)
    monkeypatch.setattr(runtime, "build_face_tracker", lambda config: object())
    monkeypatch.setattr(runtime, "build_separator", lambda config: object())
    resolver = Mock(side_effect=[6, 4])
    monkeypatch.setattr(runtime, "resolve_audio_device", resolver)
    runner = runtime.SessionRunner(
        {
            "backend": {"name": "test"},
            "audio": {"input_device": "C270", "output_device": 4},
            "video": {"device_index": 2, "fps": 30},
        },
        live=True,
    )
    runner._build()
    microphone.assert_called_once_with(16000, 1, 320, device=6)
    speaker.assert_called_once_with(16000, 1, device=4)
    camera.assert_called_once_with(device_index=2, width=640, height=480, fps=30.0)
    microphone.return_value.start.assert_not_called()
    camera.return_value.start.assert_not_called()

def test_input_recording_identifies_selected_microphone():
    runner = runtime.SessionRunner({"audio": {"input_device": "C270"}})
    runner._input_device = 6
    seen = []
    runner.recorder.on_input = seen.append
    runner._on_input(audio())
    assert seen[0].metadata["input_device"] == 6
    assert seen[0].metadata["input_device_requested"] == "C270"

def test_selection_switch_during_inference_cannot_deliver_old_output():
    sink, state, valid, _, destination, _ = prepared_sink()

    def switch_during_inference(*args):
        assert state.select("b")
        assert state.select("a")
        return valid

    separator = runtime.EpochSeparator(
        SimpleNamespace(separate=switch_during_inference),
        state,
    )
    old = separator.separate(audio(), TargetSelection(10000, track("a")), [])
    sink.write(old)
    assert not any(destination[0].data)
    assert state.snapshot().phase == "focusing"

def recordable_runner(tmp_path, monkeypatch):
    pipeline, separator = FakePipeline(), FakeSeparator()
    runner = runtime.SessionRunner(
        {},
        factory=lambda: (pipeline, separator),
        record_root=tmp_path,
    )
    ready = threading.Event()
    original = runner.state.mark_running
    monkeypatch.setattr(runner.state, "clock", lambda: 10.0)

    def mark_running_with_device_evidence():
        original()
        runner.state.observe_input(audio())
        runner.state.observe_frame(Frame(10000, None, {}))
        runner.state.observe_tracks([track("a")])
        runner.state.observe_output(audio(), {"is_real_separation": False})
        ready.set()

    monkeypatch.setattr(runner.state, "mark_running", mark_running_with_device_evidence)
    runner.start()
    try:
        assert ready.wait(timeout=5)
        assert runner.state.snapshot().phase == "listening"
    except AssertionError:
        runner.close(timeout=5)
        raise
    return runner, pipeline

def test_record_start_and_stop_run_off_ui_thread(tmp_path, monkeypatch):
    runner, _ = recordable_runner(tmp_path, monkeypatch)
    start_entered, start_release = threading.Event(), threading.Event()
    stop_entered, stop_release = threading.Event(), threading.Event()
    original_start, original_stop = (
        runner.recorder.start_clip,
        runner.recorder.stop_clip,
    )
    calls = []

    def delayed_start(*args):
        calls.append(threading.get_ident())
        start_entered.set()
        assert start_release.wait(timeout=5)
        return original_start(*args)

    def delayed_stop():
        stop_entered.set()
        assert stop_release.wait(timeout=5)
        return original_stop()

    monkeypatch.setattr(runner.recorder, "start_clip", delayed_start)
    try:
        runner.toggle_recording()
        assert start_entered.wait(timeout=5)
        assert runner.record_busy
        assert not list(tmp_path.iterdir())
        runner.toggle_recording()
        assert len(calls) == 1
        assert calls[0] != threading.get_ident()
        start_release.set()
        runner._record_thread.join(timeout=5)
        assert not runner.record_busy
        assert runner.recorder.snapshot().active
        monkeypatch.setattr(runner.recorder, "stop_clip", delayed_stop)
        runner.toggle_recording()
        assert stop_entered.wait(timeout=5)
        assert runner.record_busy
        assert runner.recorder.snapshot().active
        stop_release.set()
        runner._record_thread.join(timeout=5)
        assert not runner.record_busy
        assert not runner.recorder.snapshot().active
        assert (runner.recorder.snapshot().path / "manifest.json").exists()
    finally:
        start_release.set()
        stop_release.set()
        assert runner.close(timeout=5)

def test_session_stop_waits_for_pending_record_start_then_closes_it(
    tmp_path, monkeypatch
):
    runner, pipeline = recordable_runner(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original_start = runner.recorder.start_clip

    def delayed_start(*args):
        entered.set()
        assert release.wait(timeout=5)
        return original_start(*args)

    monkeypatch.setattr(runner.recorder, "start_clip", delayed_start)
    try:
        runner.toggle_recording()
        assert entered.wait(timeout=5)
        runner.stop()
        assert pipeline.stop_entered.wait(timeout=5)
        assert not runner.close(timeout=0.01)
        assert runner.state.snapshot().phase == "stopping"
        release.set()
        assert runner.close(timeout=5)
        assert not runner.record_busy
        assert not runner.recorder.snapshot().active
        assert runner.state.snapshot().phase == "stopped"
        assert (runner.recorder.snapshot().path / "manifest.json").exists()
    finally:
        release.set()
        assert runner.close(timeout=5)

def test_queued_record_action_rechecks_stop_before_creating_files(
    tmp_path, monkeypatch
):
    runner, pipeline = recordable_runner(tmp_path, monkeypatch)
    start = Mock(wraps=runner.recorder.start_clip)
    monkeypatch.setattr(runner.recorder, "start_clip", start)
    runner._record_io_lock.acquire()
    try:
        runner.toggle_recording()
        assert runner.record_busy
        runner.stop()
        assert pipeline.stop_entered.wait(timeout=5)
        assert not runner.close(timeout=0.01)
    finally:
        runner._record_io_lock.release()
        assert runner.close(timeout=5)
    start.assert_not_called()
    assert not runner.record_busy
    assert not runner.recorder.snapshot().active
    assert not list(tmp_path.iterdir())

def test_async_record_setup_error_remains_visible_without_ending_session(
    tmp_path,
    monkeypatch,
):
    runner, _ = recordable_runner(tmp_path, monkeypatch)
    occupied = tmp_path / "occupied"
    occupied.write_text("keep")
    runner.record_root = occupied
    try:
        runner.toggle_recording()
        runner._record_thread.join(timeout=5)
        assert not runner.record_busy
        assert "Could not start recording" in runner.recorder.snapshot().error
        assert not runner.recorder.snapshot().active
        assert runner.state.snapshot().phase == "listening"
        assert occupied.read_text() == "keep"
    finally:
        assert runner.close(timeout=5)


def test_camera_size_accepts_wxh_in_either_case() -> None:
    from demo.tap_to_select import _parse_camera_size

    assert _parse_camera_size("1280x720") == (1280, 720)
    assert _parse_camera_size("1920X1080") == (1920, 1080)
    assert _parse_camera_size("640x480") == (640, 480)


@pytest.mark.parametrize(
    "bad",
    [
        "1280",
        "1280x",
        "x720",
        "1280x720x30",
        "1280×720",
        "0x0",
        "abc",
        "10x10",
        "99999x99999",
        "-1280x720",
    ],
)
def test_camera_size_rejects_values_that_should_never_reach_a_driver(bad) -> None:
    import argparse

    from demo.tap_to_select import _parse_camera_size

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_camera_size(bad)


def _live_runner(monkeypatch, video_config):
    from onevoice.audio import io
    from onevoice.video import capture

    camera = Mock()
    monkeypatch.setattr(io, "MicrophoneSource", Mock())
    monkeypatch.setattr(io, "SpeakerSink", Mock())
    monkeypatch.setattr(capture, "WebcamSource", camera)
    monkeypatch.setattr(runtime, "build_face_tracker", lambda config: object())
    monkeypatch.setattr(runtime, "build_separator", lambda config: object())
    monkeypatch.setattr(runtime, "resolve_audio_device", Mock(side_effect=[0, 0]))
    runner = runtime.SessionRunner(
        {"backend": {"name": "test"}, "video": video_config}, live=True
    )
    return runner, camera


def test_configured_camera_size_reaches_the_webcam(monkeypatch):
    runner, camera = _live_runner(
        monkeypatch, {"device_index": 1, "width": 1280, "height": 720}
    )
    runner._build()
    camera.assert_called_once_with(device_index=1, width=1280, height=720, fps=30.0)


def test_camera_source_override_beats_config_and_untouched_keeps_it(monkeypatch):
    runner, camera = _live_runner(monkeypatch, {"device_index": 2})
    assert runner.camera_source == "external"
    assert runner.set_camera_source("built_in")
    assert runner.camera_source == "built_in"
    runner._build()
    assert camera.call_args.kwargs["device_index"] == 0


def test_unknown_camera_source_is_rejected(monkeypatch):
    runner, _ = _live_runner(monkeypatch, {})
    assert not runner.set_camera_source("hdmi")
    assert runner.camera_source == "built_in"
