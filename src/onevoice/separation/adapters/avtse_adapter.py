

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext, SeparatorAdapter
from onevoice.separation.config import SeparationConfig
from onevoice.separation.utilities import import_torch
from onevoice.video.visual_encoder.simple_encoder import (
    RollingROIBuffer,
    SimpleVisualEncoder,
    VisualEncoderConfig,
)

logger = logging.getLogger(__name__)

@dataclass
class _TrackStreamState:
    hidden: Any = None
    roi_buffer: RollingROIBuffer = field(default_factory=RollingROIBuffer)
    last_good_embed: Any = None
    last_seen_ms: float = 0.0

@dataclass
class _CallContext:
    audio_tensor: Any = None
    target: TargetSelection | None = None
    all_tracks: list[SpeakerTrack] = field(default_factory=list)
    conditioning: str = "audio_only"
    target_track_id: str | None = None

class AvtseAdapter(SeparatorAdapter):

    def __init__(self, config: SeparationConfig) -> None:
        params = config.params
        embed_dim = int(params.get("visual_embed_dim", 64))
        self._visual_max_age_ms = float(params.get("visual_max_age_ms", 120.0))
        self._encoder = SimpleVisualEncoder(
            VisualEncoderConfig(embed_dim=embed_dim)
        )
        self._track_states: dict[str, _TrackStreamState] = {}
        self._active_track_id: str | None = None
        self._call = _CallContext()
        self._logged_stale = False
        self._logged_no_visual = False

    def to_backend(
        self,
        model: Any,
        audio_chunk: AudioChunk,
        target: TargetSelection,
        all_tracks: list[SpeakerTrack],
        ctx: AdapterContext,
    ) -> Any:
        tensor = super().to_backend(model, audio_chunk, target, all_tracks, ctx)
        conditioning = "audio_only"
        target_id: str | None = None
        visual_emb: Any = None

        speaker = target.selected_speaker
        if speaker is not None:
            target_id = speaker.track_id
            self._maybe_reset_state(target_id)
            visual_emb, conditioning = self._resolve_visual(speaker, audio_chunk)
            state = self._track_states.setdefault(target_id, _TrackStreamState())
            state.last_seen_ms = time.monotonic() * 1000.0

        self._call = _CallContext(
            audio_tensor=tensor,
            target=target,
            all_tracks=all_tracks,
            conditioning=conditioning,
            target_track_id=target_id,
        )
        self._call._visual_emb = visual_emb  # type: ignore[attr-defined]
        return tensor

    def infer(self, model: Any, backend_input: Any, ctx: AdapterContext) -> Any:
        torch = import_torch()
        call = self._call
        audio = call.audio_tensor
        track_id = call.target_track_id
        visual_emb = getattr(call, "_visual_emb", None)

        if track_id is None:
            return audio

        state = self._track_states.setdefault(track_id, _TrackStreamState())
        embed_dim = int(ctx.config.params.get("visual_embed_dim", 64))
        if visual_emb is None:
            visual_emb = torch.zeros(1, embed_dim, device=ctx.device)
        else:
            visual_emb = torch.as_tensor(visual_emb, device=ctx.device).unsqueeze(0)

        hidden = state.hidden
        if hidden is not None:
            hidden = hidden.to(ctx.device)

        with torch.no_grad():
            output, new_hidden = model(audio, visual_emb, hidden)
        state.hidden = new_hidden.detach()
        return output

    def from_backend(
        self, backend_output: Any, reference: AudioChunk, ctx: AdapterContext
    ) -> AudioChunk:
        chunk = super().from_backend(backend_output, reference, ctx)
        meta = {
            **chunk.metadata,
            "conditioning": self._call.conditioning,
            "target_track_id": self._call.target_track_id,
            "fallback": False,
        }
        return AudioChunk(
            timestamp_ms=chunk.timestamp_ms,
            data=chunk.data,
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            metadata=meta,
        )

    def _maybe_reset_state(self, track_id: str) -> None:
        if self._active_track_id is not None and self._active_track_id != track_id:
            logger.info(
                "AV-TSE streaming state reset on switch %s -> %s",
                self._active_track_id,
                track_id,
            )
            self._track_states.pop(self._active_track_id, None)
        self._active_track_id = track_id

    def _resolve_visual(
        self, speaker: SpeakerTrack, audio_chunk: AudioChunk
    ) -> tuple[Any | None, str]:
        now = audio_chunk.timestamp_ms
        lip_ts = float(speaker.metadata.get("lip_roi_ts_ms", now))
        age_ms = abs(now - lip_ts)
        stale = age_ms > self._visual_max_age_ms

        patch = speaker.metadata.get("lip_patch")
        embed = self._encoder.encode(patch) if patch is not None else None
        state = self._track_states.setdefault(speaker.track_id, _TrackStreamState())

        if embed is not None and not stale:
            state.roi_buffer.push(embed)
            state.last_good_embed = embed
            return embed, "visual"

        if stale and not self._logged_stale:
            logger.warning(
                "AV-TSE visual cue stale (%.0f ms > %.0f ms); audio-only conditioning",
                age_ms,
                self._visual_max_age_ms,
            )
            self._logged_stale = True

        last = state.last_good_embed
        if last is not None and age_ms <= self._visual_max_age_ms * 2:
            return last, "stale_visual"

        if patch is None and not self._logged_no_visual:
            logger.info("AV-TSE no lip_patch in metadata; audio-only conditioning")
            self._logged_no_visual = True
        return None, "audio_only"

    def evict_stale_tracks(self, max_age_ms: float = 5000.0) -> None:
        now = time.monotonic() * 1000.0
        stale = [
            tid
            for tid, st in self._track_states.items()
            if now - st.last_seen_ms > max_age_ms
        ]
        for tid in stale:
            self._track_states.pop(tid, None)
