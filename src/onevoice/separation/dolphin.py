

from __future__ import annotations

import logging
from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter
from onevoice.separation.base import LoadableSeparator
from onevoice.separation.config import SeparationConfig
from onevoice.separation.dolphin_loader import load_dolphin_model
from onevoice.separation.utilities import chunk_to_float_list, import_numpy, make_output_chunk, silent_chunk

logger = logging.getLogger(__name__)

class DolphinSeparator(LoadableSeparator):

    def __init__(self, config: SeparationConfig) -> None:
        self._adapter_impl = DolphinAdapter(config)
        self._no_target_policy = str(
            config.params.get("no_target_policy", "silence")
        ).lower()

        self._no_target_calls = 0
        self._total_calls = 0
        super().__init__(config, self._adapter_impl)

    def _build_model(self, config: SeparationConfig, device: str) -> Any:
        return load_dolphin_model(config, device)

    def _warmup(self, model: Any, config: SeparationConfig, device: str) -> None:

        np = import_numpy()
        samples = config.chunk_size or 320
        window_s = float(config.params.get("window_s", 0.5))
        hop_s = float(config.params.get("hop_s", window_s / 2.0))
        calls_needed = int(window_s * config.sample_rate / samples) + int(
            hop_s * config.sample_rate / samples
        ) + 1
        warm_target = TargetSelection(
            0.0,
            SpeakerTrack("__warmup__", (0, 0, 1, 1), 1.0, metadata={}),
        )
        for _ in range(max(1, calls_needed)):
            chunk = silent_chunk(config.sample_rate, samples)
            self.separate(chunk, warm_target, [warm_target.selected_speaker])  # type: ignore[list-item]

        self._adapter_impl.flush_pending("__warmup__")
        self._adapter_impl._track_states.pop("__warmup__", None)  # noqa: SLF001
        with self._lock:
            self._stats["inferences"] = 0
            self._stats["total_infer_ms"] = 0.0
            self._stats["last_infer_ms"] = 0.0

    def close(self) -> None:
        self._adapter_impl.drain_and_warn_on_shutdown()
        super().close()
        self._adapter_impl.shutdown()

    def separate(
        self,
        audio_chunk: AudioChunk,
        target: TargetSelection,
        all_tracks: list[SpeakerTrack],
    ) -> AudioChunk:
        self._total_calls += 1
        if target.selected_speaker is None:
            self._no_target_calls += 1
            if self._total_calls % 100 == 0:
                logger.debug(
                    "Dolphin: %d/%d calls had no locked target (%.0f%%) — window "
                    "fill time stretches with detection gaps, not just wall-clock hop_s.",
                    self._no_target_calls,
                    self._total_calls,
                    100.0 * self._no_target_calls / self._total_calls,
                )
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
