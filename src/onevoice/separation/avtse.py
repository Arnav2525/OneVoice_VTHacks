

from __future__ import annotations

from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.avtse_adapter import AvtseAdapter
from onevoice.separation.base import LoadableSeparator
from onevoice.separation.config import SeparationConfig
from onevoice.separation.utilities import (
    chunk_to_float_list,
    import_torch,
    make_output_chunk,
    silent_chunk,
)

class AvtseSeparator(LoadableSeparator):

    def __init__(self, config: SeparationConfig) -> None:
        self._adapter_impl = AvtseAdapter(config)
        self._no_target_policy = str(
            config.params.get("no_target_policy", "silence")
        ).lower()
        super().__init__(config, self._adapter_impl)

    def _build_model(self, config: SeparationConfig, device: str) -> Any:
        torch = import_torch()
        params = config.params
        hidden = int(params.get("hidden_dim", 64))
        visual_dim = int(params.get("visual_embed_dim", 64))
        model = CausalAvtseModel(hidden_dim=hidden, visual_dim=visual_dim)
        checkpoint = config.checkpoint
        if checkpoint:
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                model.load_state_dict(state, strict=False)
            except Exception:  # noqa: BLE001 - honest degrade to random init
                pass
        model.to(device)
        model.eval()
        return model

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
                metadata={
                    **audio_chunk.metadata,
                    "conditioning": "none",
                    "target_track_id": None,
                    "no_target": True,
                },
            )
        samples = len(chunk_to_float_list(audio_chunk.data))
        silent = silent_chunk(audio_chunk.sample_rate, samples)
        return make_output_chunk(
            silent.data,
            audio_chunk,
            extra_metadata={
                "conditioning": "none",
                "target_track_id": None,
                "no_target": True,
            },
        )

class CausalAvtseModel:

    def __init__(self, hidden_dim: int = 64, visual_dim: int = 64) -> None:
        torch = import_torch()
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(1 + visual_dim, hidden_dim, batch_first=True)
                self.head = nn.Linear(hidden_dim, 1)
                self.gate = nn.Tanh()

            def forward(
                self,
                audio: Any,
                visual: Any,
                hidden: Any | None,
            ) -> tuple[Any, Any]:
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                t_len = audio.size(1)
                v = visual.unsqueeze(1).expand(-1, t_len, -1)
                x = torch.cat([audio.unsqueeze(-1), v], dim=-1)
                out, h = self.gru(x, hidden)
                y = self.head(out).squeeze(-1)
                g = self.gate(y)
                y = g * y + (1.0 - g.abs()) * audio
                return y, h

        self._module = _Model()
        self.eval = self._module.eval
        self.to = self._module.to
        self.parameters = self._module.parameters
        self.load_state_dict = self._module.load_state_dict
        self.state_dict = self._module.state_dict

    def __call__(self, audio: Any, visual: Any, hidden: Any | None) -> tuple[Any, Any]:
        out: tuple[Any, Any] = self._module(audio, visual, hidden)
        return out
