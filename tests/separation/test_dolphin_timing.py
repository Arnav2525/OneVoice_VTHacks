

from concurrent.futures import Future

import numpy as np
import pytest

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext
from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter
from onevoice.separation.config import SeparationConfig

class ImmediateExecutor:
    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        pass

@pytest.fixture
def make_adapter():
    created = []

    def make(**params):
        config = SeparationConfig(
            name="dolphin",
            device="cpu",
            sample_rate=16000,
            params={"window_s": 2.0, "hop_s": 0.75, "auto_gain": False, **params},
        )
        adapter = DolphinAdapter(config)
        adapter.shutdown()
        adapter._executor = ImmediateExecutor()
        windows = []

        def forward(model, audio, mouths, ctx, frames=None):
            windows.append((audio.copy(), np.stack(mouths)))
            return audio.copy(), True, audio.copy()

        adapter._infer_window_forward = forward
        created.append(adapter)
        return adapter, AdapterContext("cpu", config), windows

    yield make
    for adapter in created:
        adapter.shutdown()

def feed(
    adapter,
    ctx,
    audio,
    timestamp,
    frame_timestamp=None,
    frame_value=1.0,
    *,
    flush=True,
    epoch=1,
):
    metadata = {"ui_selection_epoch": epoch}
    if frame_value is not None:
        metadata["lip_patch"] = np.full((88, 88), frame_value, dtype=np.float32)
    if frame_timestamp is not None:
        metadata["lip_roi_ts_ms"] = frame_timestamp
    speaker = SpeakerTrack("person", (0, 0, 10, 10), 1.0, metadata)
    chunk = AudioChunk(timestamp, audio, 16000, 1, {})
    target = TargetSelection(timestamp, speaker)
    call = adapter.to_backend(None, chunk, target, [speaker], ctx)
    output = adapter.infer(None, call, ctx)
    result = adapter.from_backend(output, chunk, ctx)
    if flush:
        adapter.flush_pending("person")
    return result

@pytest.mark.parametrize("fps", [15, 25, 30, 60])
def test_mouth_motion_uses_same_two_seconds_as_audio(make_adapter, fps):
    adapter, ctx, windows = make_adapter()
    cursor = 0
    for index in range(2 * fps):
        end = round((index + 1) * 16000 / fps)
        timestamp = index * 1000 / fps
        feed(
            adapter,
            ctx,
            np.zeros(end - cursor),
            cursor / 16,
            timestamp,
            frame_value=timestamp + 1,
        )
        cursor = end
    assert len(windows) == 1
    _, mouths = windows[0]
    model_timestamps = np.arange(50) * 40

    np.testing.assert_allclose(mouths[:, 0, 0] - 1, model_timestamps, atol=1000 / fps)
    assert mouths.shape == (50, 88, 88)
    assert adapter._track_states["person"].last_visual_coverage == 1.0

def test_exact_windows_and_overlap_with_nondivisible_merged_chunks(make_adapter):
    adapter, ctx, windows = make_adapter()
    signal = (0.3 * np.sin(np.arange(112000) * 0.037)).astype(np.float32)
    cursor = 0
    restored = []
    sizes = [320, 640, 320, 960, 320]
    index = 0
    while cursor < len(signal):
        count = min(sizes[index % len(sizes)], len(signal) - cursor)
        result = feed(adapter, ctx, signal[cursor : cursor + count], cursor / 16)
        if result.metadata["output_ready"]:
            restored.extend(result.data)
        state = adapter._track_states["person"]
        restored.extend(adapter._pop_fifo(state, state.fifo.size))
        cursor += count
        index += 1
    for index, (audio, _) in enumerate(windows):
        start = index * 12000
        np.testing.assert_array_equal(audio, signal[start : start + 32000])
    np.testing.assert_allclose(restored, signal[: len(restored)], atol=1e-6)
    assert adapter._track_states["person"].dropped_hops == 0

def test_visual_jitter_and_duplicate_frames_do_not_shift_timeline(make_adapter):
    adapter, ctx, windows = make_adapter()
    rng = np.random.default_rng(32)
    cursor = 0
    last = None
    for index in range(60):
        end = round((index + 1) * 16000 / 30)
        timestamp = index * 1000 / 30 + rng.uniform(-7, 7)
        if index in (12, 13, 31):
            timestamp = last
        feed(
            adapter,
            ctx,
            np.zeros(end - cursor),
            cursor / 16,
            timestamp,
            frame_value=timestamp + 10,
        )
        last = timestamp
        cursor = end
    sampled = windows[0][1][:, 0, 0] - 10
    np.testing.assert_allclose(sampled, np.arange(50) * 40, atol=70)
    state = adapter._track_states["person"]
    assert state.last_visual_coverage == 1.0
    assert len(state.mouth_buf) == 57

def test_stale_visual_window_cannot_reach_output_or_next_overlap(make_adapter):
    adapter, ctx, windows = make_adapter()
    for index in range(100):
        feed(adapter, ctx, np.ones(320), index * 20, 0.0)
    state = adapter._track_states["person"]
    assert len(windows) == 1
    assert state.last_visual_coverage < 0.1
    assert state.last_conditioning == "audio_only"
    assert state.fifo.size == 0
    assert np.count_nonzero(state.ola_acc) == 0

def test_audio_gap_zero_fills_instead_of_splicing_old_audio(make_adapter):

    adapter, ctx, _ = make_adapter()
    for index in range(50):
        feed(adapter, ctx, np.ones(320), index * 20)
    old = adapter._track_states["person"]
    result = feed(adapter, ctx, np.full(320, 0.2), 1700)
    state = adapter._track_states["person"]

    assert state is old, "a repairable gap must not discard model context"
    assert result.metadata["audio_discontinuities"] == 0
    assert result.metadata["audio_resyncs"] == 1

    gap_samples = int(round(0.700 * 16000))
    assert state.resynced_samples == pytest.approx(gap_samples, abs=2)
    tail = state.audio_buf[-320:]
    np.testing.assert_allclose(tail, 0.2)
    spliced = state.audio_buf[-(320 + gap_samples) : -320]
    np.testing.assert_allclose(spliced, 0.0)
    assert state.audio_buf[-(320 + gap_samples) - 1] == pytest.approx(1.0)

def test_gap_beyond_the_resync_limit_still_rebuffers(make_adapter):

    adapter, ctx, _ = make_adapter()
    for index in range(50):
        feed(adapter, ctx, np.ones(320), index * 20)
    old = adapter._track_states["person"]
    result = feed(adapter, ctx, np.full(320, 0.2), 9000)
    state = adapter._track_states["person"]
    assert state is not old
    assert state.total_samples == 320
    assert result.metadata["audio_discontinuities"] == 1
    assert result.metadata["conditioning"] == "buffering"
    np.testing.assert_allclose(state.audio_buf, 0.2)

def test_zero_timestamp_warmup_advances_audio_sample_clock(make_adapter):
    adapter, ctx, windows = make_adapter(window_s=0.1, hop_s=0.05)
    for _ in range(5):
        feed(adapter, ctx, np.zeros(320), 0.0, frame_value=None)
    assert len(windows) == 1
    state = adapter._track_states["person"]
    assert state.total_samples == 1600
    assert state.audio_discontinuities == 0
    assert state.fifo.size == 0

def test_late_worker_drops_old_result_and_resumes_on_exact_grid(make_adapter):
    adapter, ctx, windows = make_adapter(window_s=0.1, hop_s=0.05)
    pending = Future()
    submitted = []

    class HeldExecutor(ImmediateExecutor):
        def submit(self, fn, *args):
            submitted.append(args[1].copy())
            if len(submitted) == 1:
                return pending
            return super().submit(fn, *args)

    adapter._executor = HeldExecutor()
    for index in range(25):
        feed(adapter, ctx, np.full(320, index / 100), index * 20, flush=False)
    state = adapter._track_states["person"]
    assert len(submitted) == 1
    assert state.audio_buf.size <= 3200
    assert len(state.mouth_buf) <= 32
    pending.set_result((np.ones(1600), True, np.ones(1600)))
    result = feed(adapter, ctx, np.full(320, 0.25), 500)
    assert result.metadata["output_ready"] is False
    assert state.dropped_hops >= 8
    assert state.pending_window_start == 6400
    assert state.fifo.size == 800

    assert state.fifo.max() < 0.3
    assert len(submitted) == 2

def test_output_metadata_reports_content_delay_not_hop_interval(make_adapter):
    adapter, ctx, _ = make_adapter()
    for index in range(100):
        feed(adapter, ctx, np.full(320, 0.1), index * 20)
    result = feed(adapter, ctx, np.full(320, 0.1), 2000)
    assert result.metadata["output_ready"] is True
    assert result.metadata["source_timestamp_ms"] == 0.0
    assert result.metadata["buffered_delay_ms"] == 2000.0

@pytest.mark.parametrize("gain", [-1, float("nan"), float("inf")])
def test_invalid_fixed_gain_is_rejected(gain):
    with pytest.raises(ValueError, match="output_gain"):
        DolphinAdapter(SeparationConfig(name="dolphin", params={"output_gain": gain}))

def test_fixed_gain_does_not_normalize_residual_to_loud_mixture(make_adapter):
    adapter, _, _ = make_adapter(output_gain=2.0)
    state = adapter._new_state()
    adapter._integrate_window_result(
        state, np.full(32000, 0.001), True, np.full(32000, 0.5)
    )
    np.testing.assert_allclose(state.fifo, 0.002, atol=1e-7)

def test_recent_hops_crossfade_same_samples_without_gaps_or_duplicates(make_adapter):
    adapter, ctx, _ = make_adapter(
        window_s=0.2,
        hop_s=0.05,
        output_mode="recent",
        lookahead_s=0.05,
        crossfade_s=0.01,
    )
    signal = np.random.default_rng(943).uniform(-0.2, 0.2, 12800).astype(np.float32)
    restored = []
    cursor = 0
    sizes = [320, 640, 320]
    index = 0
    while cursor < len(signal):
        count = min(sizes[index % len(sizes)], len(signal) - cursor)
        result = feed(adapter, ctx, signal[cursor : cursor + count], cursor / 16)
        if result.metadata["output_ready"]:
            restored.extend(result.data)
        state = adapter._track_states["person"]
        restored.extend(adapter._pop_fifo(state, state.fifo.size))
        cursor += count
        index += 1
    first_source = round((0.2 - 0.05 - 0.05 - 0.01) * 16000)
    np.testing.assert_allclose(
        restored, signal[first_source : first_source + len(restored)], atol=1e-7
    )
    adapter._reset_output(state)
    assert state.recent_tail is None
    assert state.fifo.size == 0

def test_recent_mode_reports_actual_content_timestamp(make_adapter):
    adapter, ctx, _ = make_adapter(
        output_mode="recent", hop_s=0.5, lookahead_s=0.5, crossfade_s=0.02
    )
    for index in range(100):
        feed(adapter, ctx, np.full(320, 0.1), index * 20)
    result = feed(adapter, ctx, np.full(320, 0.1), 2000)
    assert result.metadata["source_timestamp_ms"] == 980.0
    assert result.metadata["buffered_delay_ms"] == 1020.0

def test_recent_mode_rejects_insufficient_context():
    with pytest.raises(ValueError, match="Recent output"):
        DolphinAdapter(
            SeparationConfig(
                name="dolphin",
                params={
                    "window_s": 0.5,
                    "hop_s": 0.3,
                    "output_mode": "recent",
                    "lookahead_s": 0.3,
                },
            )
        )
