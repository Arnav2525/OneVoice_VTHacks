

from __future__ import annotations

import numpy as np

from demo.session_runtime import EpochSeparator
from demo.session_state import SessionState
from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext
from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter
from onevoice.separation.config import SeparationConfig

def test_reselecting_same_person_cannot_relabel_previous_dolphin_audio():
    config = SeparationConfig(
        name="dolphin",
        device="cpu",
        sample_rate=16000,
        params={"window_s": 1.0, "hop_s": 0.5},
    )
    adapter = DolphinAdapter(config)
    ctx = AdapterContext(device="cpu", config=config)

    class BufferedBackend:
        def separate(self, audio, target, tracks):
            call = adapter.to_backend(None, audio, target, tracks, ctx)
            output = adapter.infer(None, call, ctx)
            return adapter.from_backend(output, audio, ctx)

    state = SessionState(mode="live", clock=lambda: 100.0)
    state.begin_start()
    state.mark_running()
    person_a = SpeakerTrack("a", (0, 0, 10, 10), 0.9, {"visible": True})
    person_b = SpeakerTrack("b", (30, 0, 10, 10), 0.9, {"visible": True})
    state.observe_tracks([person_a, person_b])
    state.select("a")
    wrapper = EpochSeparator(BufferedBackend(), state)
    audio = AudioChunk(100000, [0.1] * 320, 16000, 1, {})
    target = TargetSelection(100000, person_a)
    try:
        wrapper.separate(audio, target, [person_a, person_b])
        old_state = adapter._track_states["a"]
        old_state.fifo = np.full(640, 0.25, dtype=np.float32)
        old_state.warmed_up = True
        old_state.last_conditioning = "visual"
        first = wrapper.separate(audio, target, [person_a, person_b])
        assert first.metadata["output_ready"] is True
        assert first.data == [0.25] * 320
        state.select("b")
        state.select("a")
        second = wrapper.separate(audio, target, [person_a, person_b])
        assert (
            second.metadata["ui_selection_epoch"]
            != first.metadata["ui_selection_epoch"]
        )
        assert second.metadata["ui_selection_epoch"] == state.selection()[1]
        assert second.metadata["output_ready"] is False
        assert second.data == [0.0] * 320
        assert adapter._track_states["a"] is not old_state

        assert person_a.metadata == {"visible": True}
        assert target.selected_speaker is person_a
    finally:
        adapter.shutdown()
