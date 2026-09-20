

import array

import pytest

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.frame import Frame
from onevoice.streaming.synchronizer import TimestampFrameSynchronizer

def _audio(ts):
    return AudioChunk(
        timestamp_ms=ts, data=array.array("f", [0.0]), sample_rate=16000,
        channels=1, metadata={},
    )

def _frame(ts):
    return Frame(timestamp_ms=ts, data=None, metadata={})

def test_matches_equal_timestamps():
    sync = TimestampFrameSynchronizer(sync_tolerance_ms=40.0)
    sync.push_audio(_audio(100.0))
    sync.push_video(_frame(100.0))
    chunk, frame = sync.get_synchronized_pair()
    assert chunk.timestamp_ms == 100.0
    assert frame.timestamp_ms == 100.0

def test_matches_within_tolerance():
    sync = TimestampFrameSynchronizer(sync_tolerance_ms=40.0)
    sync.push_audio(_audio(100.0))
    sync.push_video(_frame(130.0))
    chunk, frame = sync.get_synchronized_pair()
    assert abs(chunk.timestamp_ms - frame.timestamp_ms) <= 40.0
    assert sync.max_drift_ms == 30.0

def test_audio_behind_the_frame_is_carried_not_dropped():

    sync = TimestampFrameSynchronizer(sync_tolerance_ms=40.0, max_drift_ms=10_000.0)
    sync.push_audio(_audio(0.0))
    sync.push_video(_frame(500.0))
    sync.push_audio(_audio(500.0))
    chunk, frame = sync.get_synchronized_pair()
    assert frame.timestamp_ms == 500.0
    assert chunk.timestamp_ms == 0.0
    assert list(chunk.data) == [0.0, 0.0]
    assert sync.dropped_audio == 0

def test_drift_trim_drops_video_never_audio():
    sync = TimestampFrameSynchronizer(sync_tolerance_ms=40.0, max_drift_ms=100.0)
    sync.push_video(_frame(0.0))
    sync.push_audio(_audio(5000.0))
    assert sync.dropped_audio == 0
    assert sync.dropped_video == 1

def test_stalled_camera_still_releases_audio_against_last_frame():

    sync = TimestampFrameSynchronizer(
        sync_tolerance_ms=40.0, audio_release_ms=50.0, frame_hold_ms=500.0
    )
    sync.push_audio(_audio(0.0))
    sync.push_video(_frame(0.0))
    sync.get_synchronized_pair()

    for ts in (20.0, 40.0, 60.0, 80.0, 100.0):
        sync.push_audio(_audio(ts))
    chunk, frame, held = sync.get_pair_with_status()
    assert held is True
    assert frame.timestamp_ms == 0.0
    assert len(list(chunk.data)) == 5
    assert sync.dropped_audio == 0
    assert sync.held_frames == 1

def test_audio_is_not_held_against_an_ancient_frame():
    sync = TimestampFrameSynchronizer(
        sync_tolerance_ms=40.0, audio_release_ms=50.0, frame_hold_ms=100.0
    )
    sync.push_audio(_audio(0.0))
    sync.push_video(_frame(0.0))
    sync.get_synchronized_pair()

    sync.push_audio(_audio(5000.0))
    sync.push_audio(_audio(5020.0))
    with pytest.raises(TimeoutError):
        sync.get_pair_with_status()
    assert sync.dropped_audio == 0

def test_backlog_bound_is_the_only_path_that_drops_audio():
    sync = TimestampFrameSynchronizer(max_audio_backlog_ms=100.0, pair_timeout_s=0.01)
    for ts in range(0, 400, 20):
        sync.push_audio(_audio(float(ts)))
    assert sync.dropped_audio > 0
    assert sync.audio_backlog > 0

def test_timeout_when_no_pair():
    sync = TimestampFrameSynchronizer(pair_timeout_s=0.05)
    with pytest.raises(TimeoutError):
        sync.get_synchronized_pair()

def test_timeout_with_only_audio():
    sync = TimestampFrameSynchronizer(pair_timeout_s=0.05)
    sync.push_audio(_audio(10.0))
    with pytest.raises(TimeoutError):
        sync.get_synchronized_pair()

def test_multiple_audio_chunks_per_frame_are_merged_not_dropped():

    sync = TimestampFrameSynchronizer(sync_tolerance_ms=40.0)
    sync.push_audio(_audio(90.0))
    sync.push_audio(_audio(100.0))
    sync.push_audio(_audio(110.0))
    sync.push_video(_frame(100.0))
    chunk, frame = sync.get_synchronized_pair()
    assert frame.timestamp_ms == 100.0
    assert chunk.timestamp_ms == 90.0
    assert list(chunk.data) == [0.0, 0.0, 0.0]
    assert sync.dropped_audio == 0

def test_extra_audio_between_frames_is_merged_into_next_pair():

    sync = TimestampFrameSynchronizer(sync_tolerance_ms=20.0)
    sync.push_audio(_audio(0.0))
    sync.push_video(_frame(0.0))
    chunk1, frame1 = sync.get_synchronized_pair()
    assert chunk1.timestamp_ms == 0.0
    assert frame1.timestamp_ms == 0.0

    sync.push_audio(_audio(33.0))
    sync.push_audio(_audio(40.0))
    sync.push_video(_frame(40.0))
    chunk2, frame2 = sync.get_synchronized_pair()
    assert frame2.timestamp_ms == 40.0
    assert list(chunk2.data) == [0.0, 0.0]
    assert sync.dropped_audio == 0
