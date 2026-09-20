

from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pytest

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext
from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter
from onevoice.separation.config import SeparationConfig

@pytest.fixture
def adapter_context():
    config = SeparationConfig(
        name="dolphin",
        device="cpu",
        sample_rate=16000,
        params={"window_s": 1.0, "hop_s": 0.5},
    )
    adapter = DolphinAdapter(config)
    yield adapter, AdapterContext(device="cpu", config=config)
    adapter.shutdown()

def _process(adapter, ctx, *, target_id="person-1", samples=320, epoch=None):
    chunk = AudioChunk(0.0, [0.1] * samples, 16000, 1, {})
    speaker = (
        SpeakerTrack(
            target_id,
            (0, 0, 10, 10),
            0.9,
            {"ui_selection_epoch": epoch} if epoch is not None else {},
        )
        if target_id is not None
        else None
    )
    target = TargetSelection(0.0, speaker)
    call = adapter.to_backend(None, chunk, target, [], ctx)

    output = adapter.infer(None, call, ctx)
    return adapter.from_backend(output, chunk, ctx)

def _seed_output(adapter, available, value=0.0):
    state = adapter._new_state()
    state.warmed_up = True
    state.last_conditioning = "visual"
    state.fifo = np.full(available, value, dtype=np.float32)
    adapter._track_states["person-1"] = state

@pytest.mark.parametrize("available,ready", [(0, False), (100, False), (320, True)])
def test_readiness_requires_a_full_processed_chunk(adapter_context, available, ready):
    adapter, ctx = adapter_context
    _seed_output(adapter, available)
    result = _process(adapter, ctx)
    assert result.metadata["conditioning"] == "visual"
    assert result.metadata["output_ready"] is ready

    assert result.data == [0.0] * 320

def test_partial_output_is_preserved_but_not_reported_ready(adapter_context):
    adapter, ctx = adapter_context
    _seed_output(adapter, 100, value=0.25)
    result = _process(adapter, ctx)
    assert result.metadata["output_ready"] is False
    assert result.data == [0.25] * 100 + [0.0] * 220

def test_readiness_clears_after_output_is_consumed(adapter_context):
    adapter, ctx = adapter_context
    _seed_output(adapter, 320)
    assert _process(adapter, ctx).metadata["output_ready"] is True
    depleted = _process(adapter, ctx)
    assert depleted.metadata["conditioning"] == "visual"
    assert depleted.metadata["output_ready"] is False

def test_new_target_cannot_inherit_old_output_readiness(adapter_context):
    adapter, ctx = adapter_context
    _seed_output(adapter, 640)
    assert _process(adapter, ctx).metadata["output_ready"] is True
    switched = _process(adapter, ctx, target_id="person-2")
    assert switched.metadata["target_track_id"] == "person-2"
    assert switched.metadata["output_ready"] is False

def test_zero_length_chunk_is_not_output_ready(adapter_context):
    adapter, ctx = adapter_context
    _seed_output(adapter, 320)
    assert _process(adapter, ctx, samples=0).metadata["output_ready"] is False

def test_no_target_cannot_inherit_output_readiness(adapter_context):
    adapter, ctx = adapter_context
    _seed_output(adapter, 640)
    assert _process(adapter, ctx).metadata["output_ready"] is True
    result = _process(adapter, ctx, target_id=None)
    assert result.metadata["target_track_id"] is None
    assert result.metadata["output_ready"] is False

def test_same_person_new_selection_discards_buffered_output(adapter_context):
    adapter, ctx = adapter_context
    _process(adapter, ctx, epoch=1)
    old_state = adapter._track_states["person-1"]
    old_state.fifo = np.full(640, 0.25, dtype=np.float32)
    old_state.warmed_up = True
    old_state.last_conditioning = "visual"
    assert _process(adapter, ctx, epoch=1).metadata["output_ready"] is True
    reselected = _process(adapter, ctx, epoch=3)
    assert adapter._track_states["person-1"] is not old_state
    assert reselected.metadata["output_ready"] is False
    assert reselected.metadata["conditioning"] == "buffering"
    assert reselected.data == [0.0] * 320

@pytest.mark.parametrize("already_running", [False, True])
def test_old_epoch_pending_window_cannot_repopulate_new_selection(
    adapter_context,
    already_running,
):
    adapter, ctx = adapter_context
    _process(adapter, ctx, epoch=1)
    old_state = adapter._track_states["person-1"]
    old_future = Future()
    if already_running:
        assert old_future.set_running_or_notify_cancel()
    old_state.pending_future = old_future
    _process(adapter, ctx, epoch=2)
    new_state = adapter._track_states["person-1"]
    assert new_state is not old_state
    if already_running:

        waveform = np.ones(adapter._window_samples, dtype=np.float32)
        old_future.set_result((waveform, True, waveform))
    else:
        assert old_future.cancelled()
    result = _process(adapter, ctx, epoch=2)
    assert new_state.pending_future is None
    assert new_state.fifo.size == 0
    assert result.metadata["output_ready"] is False
    assert result.data == [0.0] * 320

def test_callers_without_selection_epochs_retain_same_person_buffer(adapter_context):
    adapter, ctx = adapter_context
    _process(adapter, ctx)
    original_state = adapter._track_states["person-1"]
    original_state.fifo = np.full(320, 0.25, dtype=np.float32)
    original_state.warmed_up = True
    original_state.last_conditioning = "visual"
    result = _process(adapter, ctx)
    assert adapter._track_states["person-1"] is original_state
    assert result.metadata["output_ready"] is True
    assert result.data == [0.25] * 320
