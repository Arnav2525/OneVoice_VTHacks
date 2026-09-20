

from __future__ import annotations

import array

import pytest

pytest.importorskip("numpy")

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter
from onevoice.separation.config import SeparationConfig

SR = 16000
CHUNK = 320

def _adapter(**params):
    base = {
        "window_s": 2.0,
        "hop_s": 0.5,
        "output_mode": "recent",
        "lookahead_s": 0.5,
        "crossfade_s": 0.02,
        "auto_gain": False,
    }
    base.update(params)
    return DolphinAdapter(
        SeparationConfig(
            name="dolphin", sample_rate=SR, chunk_size=CHUNK, params=base
        )
    )

def _chunk(ts_ms: float) -> AudioChunk:
    return AudioChunk(
        timestamp_ms=ts_ms,
        data=array.array("f", [0.01] * CHUNK),
        sample_rate=SR,
        channels=1,
        metadata={},
    )

def _target(track_id: str = "face-1") -> TargetSelection:
    return TargetSelection(
        0.0,
        SpeakerTrack(
            track_id=track_id,
            bounding_box=(0, 0, 10, 10),
            confidence=1.0,
            metadata={"lip_patch": None, "lip_roi_ts_ms": 0.0},
        ),
    )

def _accept(adapter, call):

    state = adapter._track_states[call.target_track_id]
    if state.origin_timestamp_ms is None:
        state.origin_timestamp_ms = call.audio_timestamp_ms
    adapter._push_audio(state, call.samples)
    if state.total_samples >= adapter._window_samples:
        state.warmed_up = True
    return state

def _feed(adapter, count, *, start_ms=0.0, step_ms=20.0, target=None):
    target = target or _target()
    ts = start_ms
    for _ in range(count):
        _accept(adapter, adapter.to_backend(None, _chunk(ts), target, [], None))
        ts += step_ms
    return ts

def test_short_gap_resyncs_and_keeps_window_state():
    adapter = _adapter()
    ts = _feed(adapter, 100)
    state = adapter._track_states["face-1"]
    before_samples = state.total_samples
    before_disc = state.audio_discontinuities

    adapter.to_backend(None, _chunk(ts + 200.0), _target(), [], None)
    state = adapter._track_states["face-1"]

    assert state.audio_discontinuities == before_disc, "must not count as a rebuild"
    assert state.audio_resyncs == 1
    assert state.resynced_samples == pytest.approx(0.200 * SR, abs=2)

    assert state.total_samples > before_samples
    assert state.warmed_up is True

def test_resync_realigns_the_sample_clock():
    adapter = _adapter()
    ts = _feed(adapter, 100)
    adapter.to_backend(None, _chunk(ts + 200.0), _target(), [], None)
    state = adapter._track_states["face-1"]
    expected = state.origin_timestamp_ms + 1000 * state.total_samples / SR

    assert expected == pytest.approx(ts + 200.0, abs=1.0)

def test_repeated_small_gaps_never_rebuild():
    adapter = _adapter()
    ts = _feed(adapter, 100)
    for _ in range(5):
        ts += 200.0
        _accept(adapter, adapter.to_backend(None, _chunk(ts), _target(), [], None))
        ts += 20.0
    state = adapter._track_states["face-1"]
    assert state.audio_discontinuities == 0
    assert state.audio_resyncs == 5

def test_huge_gap_still_rebuilds():
    adapter = _adapter()
    ts = _feed(adapter, 100)

    adapter.to_backend(None, _chunk(ts + 5000.0), _target(), [], None)
    state = adapter._track_states["face-1"]
    assert state.audio_discontinuities == 1
    assert state.warmed_up is False

def test_backwards_timestamp_rebuilds():
    adapter = _adapter()
    ts = _feed(adapter, 100)

    adapter.to_backend(None, _chunk(ts - 500.0), _target(), [], None)
    assert adapter._track_states["face-1"].audio_discontinuities == 1

def test_resync_limit_is_configurable():
    adapter = _adapter(max_resync_gap_ms=300.0)
    ts = _feed(adapter, 100)
    adapter.to_backend(None, _chunk(ts + 500.0), _target(), [], None)
    assert adapter._track_states["face-1"].audio_discontinuities == 1
