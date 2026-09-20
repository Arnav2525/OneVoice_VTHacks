

from __future__ import annotations

from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.selection.active_speaker_selector import (
    ActiveSpeakerSelector,
    LockState,
    SelectorConfig,
)

def _track(tid: str, score: float) -> SpeakerTrack:
    return SpeakerTrack(
        track_id=tid,
        bounding_box=(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        metadata={"speaking_score": score},
    )

def _frame(ts: float) -> Frame:
    return Frame(timestamp_ms=ts, data=None, metadata={})

def test_manual_override_wins():
    sel = ActiveSpeakerSelector(SelectorConfig())
    sel.set_manual_target("manual-1")
    tracks = [_track("auto-high", 0.99), _track("manual-1", 0.1)]
    out = sel.select_target(_frame(0.0), tracks)
    assert out.selected_speaker is not None
    assert out.selected_speaker.track_id == "manual-1"

def test_manual_target_coasts_briefly_when_track_flickers():

    cfg = SelectorConfig(coast_ms=200.0)
    sel = ActiveSpeakerSelector(cfg)
    sel.set_manual_target("a")
    sel.select_target(_frame(0.0), [_track("a", 0.5)])
    out = sel.select_target(_frame(50.0), [])
    assert out.selected_speaker is not None
    assert out.selected_speaker.track_id == "a"

def test_manual_target_falls_back_to_none_after_track_id_reassigned():

    cfg = SelectorConfig(coast_ms=100.0)
    sel = ActiveSpeakerSelector(cfg)
    sel.set_manual_target("a")
    sel.select_target(_frame(0.0), [_track("a", 0.5)])
    sel.select_target(_frame(50.0), [_track("c", 0.5)])
    out = sel.select_target(_frame(250.0), [_track("c", 0.5)])
    assert out.selected_speaker is None

def test_manual_target_switch_does_not_inherit_previous_stale_track():

    sel = ActiveSpeakerSelector(SelectorConfig(coast_ms=1000.0))
    sel.set_manual_target("a")
    sel.select_target(_frame(0.0), [_track("a", 0.5)])
    sel.set_manual_target("b")
    out = sel.select_target(_frame(10.0), [_track("a", 0.5)])
    assert out.selected_speaker is None

def test_acquire_after_dwell():
    cfg = SelectorConfig(acquire_thresh=0.5, acquire_dwell_ms=100.0)
    sel = ActiveSpeakerSelector(cfg)
    tracks = [_track("a", 0.8)]
    assert sel.select_target(_frame(0.0), tracks).selected_speaker is None
    assert sel.select_target(_frame(50.0), tracks).selected_speaker is None
    out = sel.select_target(_frame(120.0), tracks)
    assert out.selected_speaker is not None
    assert out.selected_speaker.track_id == "a"
    assert sel.get_state() is LockState.LOCKED

def test_coasting_when_target_lost():
    cfg = SelectorConfig(acquire_thresh=0.5, acquire_dwell_ms=0.0, coast_ms=200.0)
    sel = ActiveSpeakerSelector(cfg)
    t = _track("a", 0.9)
    sel.select_target(_frame(0.0), [t])
    out = sel.select_target(_frame(50.0), [])
    assert out.selected_speaker is not None
    assert out.selected_speaker.track_id == "a"
    assert sel.get_state() is LockState.COASTING

def test_switch_after_margin_and_dwell():
    cfg = SelectorConfig(
        acquire_thresh=0.5,
        acquire_dwell_ms=0.0,
        switch_margin=0.1,
        switch_dwell_ms=100.0,
    )
    sel = ActiveSpeakerSelector(cfg)
    sel.select_target(_frame(0.0), [_track("a", 0.6)])
    out = sel.select_target(_frame(50.0), [_track("a", 0.5), _track("b", 0.8)])
    assert out.selected_speaker is not None
    assert out.selected_speaker.track_id == "a"
    out2 = sel.select_target(_frame(200.0), [_track("a", 0.5), _track("b", 0.8)])
    assert out2.selected_speaker is not None
    assert out2.selected_speaker.track_id == "b"
