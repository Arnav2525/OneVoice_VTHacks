

from __future__ import annotations

from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.look_once_to_hear_adapter import LookOnceToHearAdapter
from onevoice.separation.base import LoadableSeparator
from onevoice.separation.config import SeparationConfig
from onevoice.separation.loth.model_loader import load_look_once_model
from onevoice.separation.utilities import (
    chunk_to_float_list,
    import_numpy,
    import_torch,
    make_output_chunk,
    silent_chunk,
)

class LookOnceToHearSeparator(LoadableSeparator):

    def __init__(self, config: SeparationConfig) -> None:
        self._adapter_impl = LookOnceToHearAdapter(config)
        self._no_target_policy = str(
            config.params.get("no_target_policy", "silence")
        ).lower()
        super().__init__(config, self._adapter_impl)

    def _build_model(self, config: SeparationConfig, device: str) -> Any:
        return load_look_once_model(config, device)

    def _warmup(self, model: Any, config: SeparationConfig, device: str) -> None:
        torch = import_torch()
        np = import_numpy()
        bundle = model if hasattr(model, "net") else model
        net = bundle.net if hasattr(bundle, "net") else bundle
        samples = config.chunk_size or 320
        stereo = np.stack(
            [np.zeros(samples, dtype=np.float32), np.zeros(samples, dtype=np.float32)]
        )
        mixture = torch.from_numpy(stereo).unsqueeze(0).to(device)
        embed = torch.zeros(1, 256, device=device)
        state = net.init_buffers(1, device)
        with torch.no_grad():
            net.predict(mixture, embed, state, pad=True)

    def separate(
        self,
        audio_chunk: AudioChunk,
        target: TargetSelection,
        all_tracks: list[SpeakerTrack],
    ) -> AudioChunk:
        if target.selected_speaker is None:
            return self._no_target_output(audio_chunk)
        return super().separate(audio_chunk, target, all_tracks)

    def _no_target_output(self, audio_chunk: AudioChunk) -> AudioChunk:
        if self._no_target_policy == "passthrough":
            return AudioChunk(
                timestamp_ms=audio_chunk.timestamp_ms,
                data=audio_chunk.data,
                sample_rate=audio_chunk.sample_rate,
                channels=audio_chunk.channels,
                metadata={**audio_chunk.metadata, "no_target": True},
            )
        samples = len(chunk_to_float_list(audio_chunk.data))
        silent = silent_chunk(audio_chunk.sample_rate, samples)
        return make_output_chunk(
            silent.data,
            audio_chunk,
            extra_metadata={"no_target": True, "conditioning": "none"},
        )
