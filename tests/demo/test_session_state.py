

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from demo.session_state import SessionState

REAL_BACKEND = {"is_real_separation": True, "fallback_active": False}

class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds=0.02):
        self.now += seconds

def _track(track_id, *, visible=True):
    return SimpleNamespace(track_id=track_id, metadata={"visible": visible})

def _audio(clock, *, timestamp=None, data=None, metadata=None):
    return SimpleNamespace(
        timestamp_ms=clock() * 1000 if timestamp is None else timestamp,
        data=[0.25] * 320 if data is None else data,
        metadata={} if metadata is None else metadata,
    )

def _frame(clock, *, timestamp=None):
    return SimpleNamespace(
        timestamp_ms=clock() * 1000 if timestamp is None else timestamp,
        data=np.zeros((20, 20, 3), dtype=np.uint8),
    )

def _output(state, clock, *, target=None, epoch=None, data=None, **metadata):
    selected, current_epoch = state.selection()
    return _audio(
        clock,
        data=data,
        metadata={
            "ui_selection_epoch": current_epoch if epoch is None else epoch,
            "target_track_id": selected if target is None else target,
            "conditioning": "visual",
            "output_ready": True,
            "fallback": False,
            "passthrough": False,
            **metadata,
        },
    )

def _refresh(state, clock, *, skip=None):
    if skip != "audio":
        state.observe_input(_audio(clock))
    if skip != "video":
        state.observe_frame(_frame(clock))
    if skip != "tracks":
        state.observe_tracks([_track("a"), _track("b"), _track("c")])
    if skip != "playback":
        state.observe_output(_output(state, clock), REAL_BACKEND)

@pytest.fixture
def running():
    clock = Clock()
    state = SessionState(mode="live", clock=clock)
    state.begin_start()
    state.mark_running()
    _refresh(state, clock)
    return state, clock

def test_ready_start_and_listening_require_fresh_device_data():
    clock = Clock()
    state = SessionState(mode="live", clock=clock)
    assert state.snapshot().phase == "ready"
    assert not state.snapshot().active
    assert not state.select("a")
    state.begin_start()
    assert state.snapshot().phase == "starting"
    assert state.snapshot().active
    state.mark_running()
    assert state.snapshot().phase == "starting"
    state.observe_input(_audio(clock))
    state.observe_frame(_frame(clock))
    state.observe_tracks([_track("a")])
    assert state.snapshot().phase == "starting"
    state.observe_output(_output(state, clock), REAL_BACKEND)
    view = state.snapshot()
    assert view.phase == "listening"
    assert view.title == "Select someone to hear"
    assert view.input_level == pytest.approx(0.25)
    assert view.output_level == 0.0

@pytest.mark.parametrize("data", [[], [float("nan")], [float("inf")], ["invalid"]])
def test_invalid_output_is_never_allowed_or_shown_as_hearing(running, data):
    state, clock = running
    state.select("a")
    output = _output(state, clock, data=data)
    assert not state.output_allowed(output)
    state.observe_output(output, REAL_BACKEND)
    assert state.snapshot().phase == "focusing"
    assert state.snapshot().output_level == 0.0

def test_selection_focuses_until_processed_output_arrives(running):
    state, clock = running
    assert state.select("a")
    view = state.snapshot()
    assert view.phase == "focusing"
    assert view.person == "Person 1"
    assert "Person 1" in view.title
    state.observe_output(_output(state, clock), REAL_BACKEND)
    assert state.snapshot().phase == "isolating"
    assert state.snapshot().title == "Hearing Person 1"
    assert state.snapshot().output_level == pytest.approx(0.25)

def test_processed_silence_still_counts_as_working_isolation(running):
    state, clock = running
    state.select("a")
    state.observe_output(_output(state, clock, data=[0.0] * 320), REAL_BACKEND)
    assert state.snapshot().phase == "isolating"
    assert state.snapshot().output_level == 0.0

@pytest.mark.parametrize(
    "metadata",
    [
        {"output_ready": False},
        {"output_ready": None},
        {"conditioning": "buffering"},
        {"conditioning": "audio_only"},
        {"fallback": True},
        {"passthrough": True},
        {"target_track_id": "b"},
        {"ui_selection_epoch": -1},
    ],
)
def test_unproven_or_wrong_target_output_never_turns_green(running, metadata):
    state, clock = running
    state.select("a")
    state.observe_output(_output(state, clock), REAL_BACKEND)
    assert state.snapshot().phase == "isolating"
    output = _output(state, clock, **metadata)
    assert not state.output_allowed(output)
    state.observe_output(output, REAL_BACKEND)
    assert state.snapshot().phase != "isolating"
    assert state.snapshot().output_level == 0.0

@pytest.mark.parametrize(
    "backend",
    [
        {"is_real_separation": False},
        {"is_real_separation": True, "fallback_active": True},
        {},
    ],
)
def test_backend_must_confirm_real_separation(running, backend):
    state, clock = running
    state.select("a")
    state.observe_output(_output(state, clock), backend)
    assert state.snapshot().phase == "unavailable"
    assert state.snapshot().title == "Voice isolation unavailable"
    assert state.snapshot().output_level == 0.0
    state.observe_output(_output(state, clock), REAL_BACKEND)
    assert state.snapshot().phase == "isolating"

def test_preview_always_mutes_output_even_with_real_backend_metadata():
    clock = Clock()
    state = SessionState(mode="preview", clock=clock)
    state.begin_start()
    state.mark_running()
    _refresh(state, clock)
    assert state.select("a")
    output = _output(state, clock)
    assert not state.output_allowed(output)
    state.observe_output(output, REAL_BACKEND)
    assert state.snapshot().phase == "listening"
    assert "Preview" in state.snapshot().detail
    assert state.snapshot().output_level == 0.0

def test_a_to_b_to_a_does_not_accept_first_selection_audio(running):
    state, clock = running
    state.select("a")
    old_a = _output(state, clock)
    state.select("b")
    state.select("a")
    assert not state.output_allowed(old_a)
    state.observe_output(old_a, REAL_BACKEND)
    assert state.snapshot().phase == "focusing"
    state.observe_output(_output(state, clock), REAL_BACKEND)
    assert state.snapshot().phase == "isolating"

@pytest.mark.parametrize("missing", ["audio", "video", "tracks", "playback"])
def test_every_required_stream_must_stay_fresh(running, missing):
    state, clock = running
    state.select("a")
    state.observe_output(_output(state, clock), REAL_BACKEND)
    clock.advance(2.01)
    _refresh(state, clock, skip=missing)
    assert state.snapshot().phase == "interrupted"
    assert state.snapshot().output_level == 0.0
    _refresh(state, clock)
    assert state.snapshot().phase == "isolating"

@pytest.mark.parametrize("coasting", [False, True])
def test_lost_target_stays_lost_until_explicit_reselection(running, coasting):
    state, clock = running
    state.select("a")
    state.observe_output(_output(state, clock), REAL_BACKEND)
    remaining = [_track("b"), _track("c")]
    if coasting:
        remaining.append(_track("a", visible=False))
    state.observe_tracks(remaining)
    assert state.snapshot().phase == "target_lost"
    assert state.selection()[0] is None
    assert not state.select("a")
    state.observe_tracks([_track("a"), _track("b"), _track("c")])
    assert state.snapshot().phase == "target_lost"
    assert not state.output_allowed(_output(state, clock, target="a"))
    assert state.select("a")
    assert state.snapshot().phase == "focusing"

def test_three_people_have_stable_distinct_labels_and_can_each_be_selected(running):
    state, _ = running
    assert state.snapshot().face_count == 3
    for track_id, name in [("a", "Person 1"), ("b", "Person 2"), ("c", "Person 3")]:
        assert state.person_name(track_id) == name
        assert state.select(track_id)
        assert state.snapshot().person == name
    state.observe_tracks([_track("c"), _track("a"), _track("b")])
    assert state.person_name("c") == "Person 3"
    assert state.select(None)
    assert state.snapshot().phase == "listening"

def test_stop_rejects_late_callbacks_and_restart_clears_selection(running):
    state, clock = running
    state.select("a")
    old_output = _output(state, clock)
    state.observe_output(old_output, REAL_BACKEND)
    state.begin_stop()
    assert state.snapshot().phase == "stopping"
    assert not state.output_allowed(old_output)
    _refresh(state, clock)
    assert state.snapshot().phase == "stopping"
    state.mark_stopped()
    _refresh(state, clock)
    view = state.snapshot()
    assert view.phase == "stopped"
    assert view.title == "Session stopped"
    assert not view.active
    assert view.face_count == 0
    assert view.input_level == view.output_level == 0.0
    clock.advance()
    state.begin_start()
    state.mark_running()
    assert state.selection()[0] is None
    assert state.snapshot().face_count == 0
    _refresh(state, clock)
    assert state.select("a")
    assert not state.output_allowed(old_output)
    state.observe_output(old_output, REAL_BACKEND)
    assert state.snapshot().phase == "focusing"

@pytest.mark.parametrize("stream", ["audio", "video"])
def test_repeating_capture_timestamp_cannot_keep_session_alive(running, stream):
    state, clock = running
    state.select("a")
    old_timestamp = clock() * 1000
    clock.advance(2.01)
    _refresh(state, clock, skip=stream)
    if stream == "audio":
        state.observe_input(_audio(clock, timestamp=old_timestamp))
    else:
        state.observe_frame(_frame(clock, timestamp=old_timestamp))
    assert state.snapshot().phase == "interrupted"

@pytest.mark.parametrize("stream", ["audio", "video"])
def test_backwards_capture_timestamp_cannot_refresh_stale_stream(running, stream):
    state, clock = running
    state.select("a")
    old_timestamp = clock() * 1000 - 20
    clock.advance(2.01)
    _refresh(state, clock, skip=stream)
    if stream == "audio":
        state.observe_input(_audio(clock, timestamp=old_timestamp))
    else:
        state.observe_frame(_frame(clock, timestamp=old_timestamp))
    assert state.snapshot().phase == "interrupted"

def test_capture_from_previous_session_cannot_complete_restart(running):
    state, clock = running
    old_audio, old_frame = _audio(clock), _frame(clock)
    state.begin_stop()
    state.mark_stopped()
    clock.advance(1.0)
    state.begin_start()
    state.mark_running()
    state.observe_input(old_audio)
    state.observe_frame(old_frame)
    state.observe_tracks([_track("a")])
    state.observe_output(_output(state, clock), REAL_BACKEND)
    assert state.snapshot().phase == "starting"

@pytest.mark.parametrize(
    "invalid", [[], [float("nan")], [float("inf")], None, "invalid"]
)
def test_invalid_input_data_cannot_refresh_microphone_health(running, invalid):
    state, clock = running
    clock.advance(2.01)
    _refresh(state, clock, skip="audio")
    chunk = _audio(clock)
    chunk.data = invalid
    state.observe_input(chunk)
    view = state.snapshot()
    assert view.phase == "interrupted"
    assert view.title == "Microphone interrupted"
    assert view.input_level == 0.0

def test_real_silent_capture_keeps_microphone_healthy(running):
    state, clock = running
    clock.advance(2.01)
    _refresh(state, clock, skip="audio")
    state.observe_input(_audio(clock, data=[0.0] * 320))
    assert state.snapshot().phase == "listening"
    assert state.snapshot().input_level == 0.0

@pytest.mark.parametrize(
    "mode,expected", [("live", "starting"), ("preview", "listening")]
)
def test_none_frame_only_counts_in_preview_mode(mode, expected):
    clock = Clock()
    state = SessionState(mode=mode, clock=clock)
    state.begin_start()
    state.mark_running()
    state.observe_input(_audio(clock))
    frame = _frame(clock)
    frame.data = None
    state.observe_frame(frame)
    state.observe_tracks([_track("a")])
    state.observe_output(_output(state, clock), REAL_BACKEND)
    assert state.snapshot().phase == expected

@pytest.mark.parametrize("invalid_stamp", [float("nan"), float("inf"), 110000.0])
def test_invalid_or_future_source_timestamps_cannot_refresh_health(
    running, invalid_stamp
):
    state, clock = running
    clock.advance(2.01)
    state.observe_input(_audio(clock, timestamp=invalid_stamp))
    state.observe_frame(_frame(clock, timestamp=invalid_stamp))
    state.observe_tracks([_track("a")])
    state.observe_output(_output(state, clock), REAL_BACKEND)
    assert state.snapshot().phase == "interrupted"

def test_replayed_output_timestamp_cannot_keep_audio_output_healthy(running):
    state, clock = running
    state.select("a")
    output = _output(state, clock)
    state.observe_output(output, REAL_BACKEND)
    clock.advance(2.01)
    _refresh(state, clock, skip="playback")
    assert not state.output_allowed(output)
    state.observe_output(output, REAL_BACKEND)
    assert state.snapshot().phase == "interrupted"
    assert state.snapshot().title == "Audio output interrupted"
    assert state.snapshot().output_level == 0.0
    state.observe_output(_output(state, clock), REAL_BACKEND)
    assert state.snapshot().phase == "isolating"

def test_pre_restart_output_timestamp_is_rejected_even_with_current_epoch(running):
    state, clock = running
    old_stamp = clock() * 1000
    state.begin_stop()
    state.mark_stopped()
    clock.advance(0.5)
    state.begin_start()
    state.mark_running()
    _refresh(state, clock)
    state.select("a")
    old_output = _output(state, clock)
    old_output.timestamp_ms = old_stamp
    assert not state.output_allowed(old_output)
    state.observe_output(old_output, REAL_BACKEND)
    assert state.snapshot().phase == "focusing"

def test_capture_during_startup_remains_valid_after_mark_running():
    clock = Clock()
    state = SessionState(mode="live", clock=clock)
    state.begin_start()
    clock.advance(0.02)
    audio, frame = _audio(clock), _frame(clock)
    clock.advance(0.02)
    state.mark_running()
    state.observe_input(audio)
    state.observe_frame(frame)
    state.observe_tracks([_track("a")])
    state.observe_output(_output(state, clock), REAL_BACKEND)
    assert state.snapshot().phase == "listening"

def test_failed_cleanup_can_return_to_stopping_for_retry(running):
    state, clock = running
    state.begin_stop()
    state.mark_stopped("Device cleanup failed")
    assert state.snapshot().phase == "error"
    state.begin_stop()
    assert state.snapshot().phase == "stopping"
    assert not state.output_allowed(_output(state, clock))
    state.mark_stopped()
    assert state.snapshot().phase == "stopped"
