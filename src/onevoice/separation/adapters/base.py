

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.config import SeparationConfig
from onevoice.separation.utilities import (
    chunk_to_float_list,
    fit_length,
    import_numpy,
    import_torch,
    make_output_chunk,
    resample_waveform,
)

@dataclass(frozen=True)
class AdapterContext:

    device: str
    config: SeparationConfig

class SeparatorAdapter:

    def to_backend(
        self,
        model: Any,
        audio_chunk: AudioChunk,
        target: TargetSelection,
        all_tracks: list[SpeakerTrack],
        ctx: AdapterContext,
    ) -> Any:
        torch = import_torch()
        np = import_numpy()
        samples = chunk_to_float_list(audio_chunk.data)
        wav = np.asarray(samples, dtype=np.float32)
        if audio_chunk.channels > 1 and wav.size:
            wav = wav.reshape(-1, audio_chunk.channels).mean(axis=1)
        model_rate = ctx.config.sample_rate
        if model_rate > 0 and audio_chunk.sample_rate != model_rate:
            if not ctx.config.resample:
                raise ValueError(
                    f"unsupported sample rate {audio_chunk.sample_rate} Hz: backend "
                    f"'{ctx.config.name}' expects {model_rate} Hz "
                    f"(set backend.resample: true to auto-resample)"
                )
            wav = resample_waveform(wav, audio_chunk.sample_rate, model_rate)
        wav = np.ascontiguousarray(wav, dtype=np.float32)
        tensor = torch.from_numpy(wav).to(ctx.device)
        return tensor.unsqueeze(0)

    def infer(self, model: Any, backend_input: Any, ctx: AdapterContext) -> Any:
        raise NotImplementedError

    def from_backend(
        self, backend_output: Any, reference: AudioChunk, ctx: AdapterContext
    ) -> AudioChunk:
        tensor = self._to_1d(backend_output)
        wav = tensor.detach().to("cpu").float().numpy()
        model_rate = ctx.config.sample_rate
        if (
            model_rate > 0
            and reference.sample_rate != model_rate
            and ctx.config.resample
        ):

            wav = resample_waveform(wav, model_rate, reference.sample_rate)
            frames = len(chunk_to_float_list(reference.data)) // max(
                1, reference.channels
            )
            wav = fit_length(wav, frames)
        samples = wav.reshape(-1).tolist()
        return make_output_chunk(
            samples, reference, extra_metadata={"backend": ctx.config.name}
        )

    def select_source(
        self, sources: Any, target: TargetSelection, ctx: AdapterContext
    ) -> Any:

        index = 0
        speaker = target.selected_speaker
        if speaker is not None:
            index = int(speaker.metadata.get("source_index", 0))
        arr = sources
        if hasattr(arr, "dim"):
            if arr.dim() == 3:
                arr = arr[0]
            if arr.dim() == 2:
                index = min(index, arr.shape[0] - 1)
                return arr[index]
        return arr

    @staticmethod
    def _to_1d(tensor: Any) -> Any:
        arr = tensor
        while hasattr(arr, "dim") and arr.dim() > 1:
            arr = arr[0]
        return arr
