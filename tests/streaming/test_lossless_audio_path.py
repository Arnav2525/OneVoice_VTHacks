

from __future__ import annotations

import array

import pytest

pytest.importorskip("numpy")
import numpy as np

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter
from onevoice.separation.config import SeparationConfig
from onevoice.streaming.synchronizer import TimestampFrameSynchronizer

SR = 16000
CHUNK_MS = 20.0
CHUNK = int(SR * CHUNK_MS / 1000)
FRAME_MS = 1000.0 / 30.0

def _audio(ts_ms, value=0.1):
    return AudioChunk(
        timestamp_ms=ts_ms,
        data=array.array("f", [value] * CHUNK),
        sample_rate=SR,
        channels=1,
        metadata={},
    )

def _target(ts_ms):
    return TargetSelection(
        ts_ms,
        SpeakerTrack(
            track_id="face-20",
            bounding_box=(0, 0, 10, 10),
            confidence=1.0,
            metadata={"lip_patch": None, "lip_roi_ts_ms": ts_ms},
        ),
    )

def _run(duration_ms=6000.0, stall_from=2000.0, stall_ms=400.0):

    sync = TimestampFrameSynchronizer(sync_tolerance_ms=40.0, pair_timeout_s=0.0)
    adapter = DolphinAdapter(
        SeparationConfig(
            name="dolphin",
            sample_rate=SR,
            chunk_size=CHUNK,
            params={"window_s": 2.0, "hop_s": 0.5, "auto_gain": False},
        )
    )

    audio_times = [t * CHUNK_MS for t in range(int(duration_ms / CHUNK_MS))]

    video_times = [
        i * FRAME_MS
        for i in range(int(duration_ms / FRAME_MS))
        if not (stall_from <= i * FRAME_MS < stall_from + stall_ms)
    ]

    pushed = 0
    pairs = []
    vi = 0
    for ts in audio_times:
        while vi < len(video_times) and video_times[vi] <= ts:
            sync.push_video(
                Frame(timestamp_ms=video_times[vi], data=None, metadata={})
            )
            vi += 1
        sync.push_audio(_audio(ts))
        pushed += 1
        while True:
            try:
                pairs.append(sync.get_pair_with_status())
            except TimeoutError:
                break
    return sync, adapter, pushed, pairs

def test_no_audio_is_lost_across_a_camera_stall():
    sync, _, pushed, pairs = _run()
    delivered = sum(len(list(chunk.data)) for chunk, _frame, _held in pairs)

    assert sync.dropped_audio == 0
    assert delivered + sync.audio_backlog * CHUNK == pushed * CHUNK
    assert sync.held_frames > 0, "the stall must have exercised the hold path"

def test_adapter_never_rebuilds_its_window_across_the_stall():
    _sync, adapter, _pushed, pairs = _run()
    for chunk, _frame, _held in pairs:
        call = adapter.to_backend(
            None, chunk, _target(chunk.timestamp_ms), [], None
        )
        state = adapter._track_states[call.target_track_id]
        if state.origin_timestamp_ms is None:
            state.origin_timestamp_ms = call.audio_timestamp_ms
        adapter._push_audio(state, call.samples)

    state = adapter._track_states["face-20"]
    assert state.audio_discontinuities == 0, (
        "a camera stall must never cost a window rebuild; that is the ~2.2 s "
        "silent hole seen in the live recordings"
    )

def test_audio_timeline_stays_contiguous_and_monotonic():
    _sync, _adapter, _pushed, pairs = _run()
    timestamps = [chunk.timestamp_ms for chunk, _f, _h in pairs]
    assert timestamps == sorted(timestamps)
    for (chunk, _f, _h), next_ts in zip(pairs, timestamps[1:]):
        span_ms = len(list(chunk.data)) / SR * 1000.0

        assert next_ts == pytest.approx(chunk.timestamp_ms + span_ms, abs=1e-6)

def test_held_pairs_are_flagged_so_perception_is_not_rerun():
    _sync, _adapter, _pushed, pairs = _run()
    held = [p for p in pairs if p[2]]
    assert held, "expected at least one held pair during the stall"

    for chunk, _frame, _ in held:
        assert len(list(chunk.data)) > 0
    assert np is not None
