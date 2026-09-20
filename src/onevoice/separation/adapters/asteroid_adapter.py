

from __future__ import annotations

from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext, SeparatorAdapter
from onevoice.separation.utilities import import_torch

class AsteroidAdapter(SeparatorAdapter):
    def __init__(self) -> None:
        self._last_target: TargetSelection | None = None

    def to_backend(self, model, audio_chunk, target, all_tracks, ctx):  # type: ignore[no-untyped-def]
        self._last_target = target
        return super().to_backend(model, audio_chunk, target, all_tracks, ctx)

    def infer(self, model: Any, backend_input: Any, ctx: AdapterContext) -> Any:
        torch = import_torch()
        with torch.no_grad():
            est = model(backend_input)
        return est[0]

    def from_backend(
        self, backend_output: Any, reference: AudioChunk, ctx: AdapterContext
    ) -> AudioChunk:
        target = self._last_target
        if target is not None:
            backend_output = self.select_source(backend_output, target, ctx)
        return super().from_backend(backend_output, reference, ctx)
