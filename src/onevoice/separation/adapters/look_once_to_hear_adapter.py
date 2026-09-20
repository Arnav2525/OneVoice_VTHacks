

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext, SeparatorAdapter
from onevoice.separation.config import SeparationConfig
from onevoice.separation.loth.model_loader import LothModels
from onevoice.separation.utilities import (
    chunk_to_float_list,
    import_numpy,
    import_torch,
    resample_waveform,
)

logger = logging.getLogger(__name__)

@dataclass
class _TrackState:
    enroll_buffer: list[float] = field(default_factory=list)
    embedding: Any | None = None
    stream_state: Any | None = None
    enrolled: bool = False

@dataclass
class _CallContext:
    mixture: Any = None
    enrolled: bool = False
    enrolling: bool = False
    target_track_id: str | None = None
    conditioning: str = "none"

class LookOnceToHearAdapter(SeparatorAdapter):

    def __init__(self, config: SeparationConfig) -> None:
        params = config.params
        self._enroll_seconds = float(params.get("enrollment_seconds", 3.0))
        self._sample_rate = int(config.sample_rate or 16000)
        self._min_enroll_samples = max(
            1, int(self._enroll_seconds * self._sample_rate)
        )
        self._track_states: dict[str, _TrackState] = {}
        self._active_track_id: str | None = None
        self._call = _CallContext()
        self._logged_enrolling = False

    def _models(self, model: Any) -> LothModels:
        if isinstance(model, LothModels):
            return model
        return LothModels(net=model, embed_net=model)

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
        bundle = self._models(model)
        samples = chunk_to_float_list(audio_chunk.data)
        wav = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio_chunk.channels > 1 and wav.size:
            wav = wav.reshape(-1, audio_chunk.channels).mean(axis=1)
        model_rate = ctx.config.sample_rate
        if (
            model_rate > 0
            and audio_chunk.sample_rate != model_rate
            and ctx.config.resample
        ):
            wav = resample_waveform(wav, audio_chunk.sample_rate, model_rate)
        wav = np.ascontiguousarray(wav, dtype=np.float32)

        speaker = target.selected_speaker
        conditioning = "none"
        enrolled = False
        enrolling = False
        target_id: str | None = None
        embedding: Any | None = None

        if speaker is not None:
            target_id = speaker.track_id
            self._maybe_reset_track(target_id)
            state = self._track_states.setdefault(target_id, _TrackState())
            state.enroll_buffer.extend(wav.tolist())
            if not state.enrolled and len(state.enroll_buffer) >= self._min_enroll_samples:
                state.embedding = self._compute_embedding(
                    state.enroll_buffer, bundle.embed_net, ctx.device
                )
                state.enrolled = True
                state.stream_state = bundle.net.init_buffers(1, ctx.device)
                logger.info(
                    "LookOnceToHear enrolled track %s (%.1fs)",
                    target_id,
                    len(state.enroll_buffer) / self._sample_rate,
                )
            if state.enrolled and state.embedding is not None:
                embedding = state.embedding.to(ctx.device)
                enrolled = True
                conditioning = "binaural_embed"
            else:
                enrolling = True
                conditioning = "enrolling"
                if not self._logged_enrolling:
                    logger.info(
                        "LookOnceToHear collecting enrollment audio (%.1fs needed)",
                        self._enroll_seconds,
                    )
                    self._logged_enrolling = True

        stereo = np.stack([wav, wav], axis=0)
        mixture = torch.from_numpy(stereo).unsqueeze(0).to(ctx.device)

        stream_state = None
        if speaker is not None:
            stream_state = state.stream_state
            if stream_state is None and enrolled:
                stream_state = bundle.net.init_buffers(1, ctx.device)
                state.stream_state = stream_state

        self._call = _CallContext(
            mixture=mixture,
            enrolled=enrolled,
            enrolling=enrolling,
            target_track_id=target_id,
            conditioning=conditioning,
        )
        self._call._embedding = embedding  # type: ignore[attr-defined]
        self._call._stream_state = stream_state  # type: ignore[attr-defined]
        self._call._net = bundle.net  # type: ignore[attr-defined]
        return mixture

    def infer(self, model: Any, backend_input: Any, ctx: AdapterContext) -> Any:
        torch = import_torch()
        call = self._call
        net = getattr(call, "_net", self._models(model).net)
        if not call.enrolled:
            return call.mixture[:, 0, :]
        embedding = getattr(call, "_embedding", None)
        stream_state = getattr(call, "_stream_state", None)
        if embedding is None or stream_state is None:
            return call.mixture[:, 0, :]
        with torch.no_grad():
            output, new_state = net.predict(
                call.mixture, embedding, stream_state, pad=True
            )
        if call.target_track_id is not None:
            st = self._track_states.get(call.target_track_id)
            if st is not None:
                st.stream_state = new_state
        if output.dim() == 3:
            return output[:, 0, :]
        return output

    def from_backend(
        self, backend_output: Any, reference: AudioChunk, ctx: AdapterContext
    ) -> AudioChunk:
        chunk = super().from_backend(backend_output, reference, ctx)
        meta = {
            **chunk.metadata,
            "conditioning": self._call.conditioning,
            "target_track_id": self._call.target_track_id,
            "enrolling": self._call.enrolling,
            "enrolled": self._call.enrolled,
            "fallback": False,
        }
        return AudioChunk(
            timestamp_ms=chunk.timestamp_ms,
            data=chunk.data,
            sample_rate=chunk.sample_rate,
            channels=chunk.channels,
            metadata=meta,
        )

    def _maybe_reset_track(self, track_id: str) -> None:
        if self._active_track_id is not None and self._active_track_id != track_id:
            logger.info(
                "LookOnceToHear reset on switch %s -> %s",
                self._active_track_id,
                track_id,
            )
            self._track_states.pop(self._active_track_id, None)
        self._active_track_id = track_id

    def _compute_embedding(
        self, samples: list[float], embed_net: Any, device: str
    ) -> Any:
        torch = import_torch()
        np = import_numpy()
        wav = np.asarray(samples, dtype=np.float32)
        stereo = np.stack([wav, wav], axis=0)
        enroll = torch.from_numpy(stereo).unsqueeze(0).to(device)
        with torch.no_grad():
            return embed_net(enroll)
