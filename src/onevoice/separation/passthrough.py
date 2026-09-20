

from __future__ import annotations

from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection

class PassthroughSeparator:

    def __init__(self, config: Any | None = None) -> None:
        self._config = config

    def get_status(self) -> dict[str, Any]:

        name = "passthrough"
        cfg = self._config
        if cfg is not None:
            cfg_name = getattr(cfg, "name", None)
            if cfg_name is None and isinstance(cfg, dict):
                cfg_name = cfg.get("name")
            name = cfg_name or "passthrough"
        return {
            "backend": name,
            "device": "cpu",
            "precision": "float32",
            "loaded": True,
            "is_real_separation": False,
            "healthy": True,
            "fallback_active": False,
            "fallback_on_error": False,
            "inferences": 0,
            "failures": 0,
            "fallbacks": 0,
            "avg_infer_ms": 0.0,
            "last_infer_ms": 0.0,
            "peak_memory_mb": 0.0,
            "load_time_ms": 0.0,
            "warmup_time_ms": 0.0,
            "last_error": None,
        }

    def separate(
        self,
        audio_chunk: AudioChunk,
        target: TargetSelection,
        all_tracks: list[SpeakerTrack],
    ) -> AudioChunk:
        return AudioChunk(
            timestamp_ms=audio_chunk.timestamp_ms,
            data=audio_chunk.data,
            sample_rate=audio_chunk.sample_rate,
            channels=audio_chunk.channels,
            metadata={**audio_chunk.metadata, "passthrough": True},
        )
