

from __future__ import annotations

from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext, SeparatorAdapter
from onevoice.separation.utilities import import_torch

class RavenAdapter(SeparatorAdapter):
    def to_backend(
        self,
        model: Any,
        audio_chunk: AudioChunk,
        target: TargetSelection,
        all_tracks: list[SpeakerTrack],
        ctx: AdapterContext,
    ) -> Any:
        mixture = super().to_backend(model, audio_chunk, target, all_tracks, ctx)
        target_feature = self._target_feature(target, ctx)
        return {"mixture": mixture, "target": target_feature}

    def infer(self, model: Any, backend_input: Any, ctx: AdapterContext) -> Any:
        torch = import_torch()
        mixture = backend_input["mixture"]
        target_feature = backend_input["target"]
        with torch.no_grad():
            if target_feature is not None:
                return model(mixture, target_feature)
            return model(mixture)

    def _target_feature(self, target: TargetSelection, ctx: AdapterContext) -> Any:
        speaker = target.selected_speaker
        if speaker is None:
            return None
        embedding = speaker.metadata.get("embedding")
        if embedding is None:
            return None
        torch = import_torch()
        tensor = torch.as_tensor(embedding, dtype=torch.float32).to(ctx.device)
        return tensor.unsqueeze(0)
