

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext, SeparatorAdapter
from onevoice.separation.config import SeparationConfig
from onevoice.separation.loader import BackendLoader
from onevoice.separation.utilities import (
    empty_target,
    memory_used_mb,
    run_with_timeout,
    silent_chunk,
)

logger = logging.getLogger(__name__)

class LoadableSeparator:

    def __init__(self, config: SeparationConfig, adapter: SeparatorAdapter) -> None:
        self._config = config
        self._adapter = adapter
        self._loader = BackendLoader(config, self._build_model, self._warmup)
        self._timeout_s = config.timeout_ms / 1000.0
        self._lock = threading.Lock()
        self._fallback_active = False
        self._last_error: str | None = None
        self._warn_every = 100
        self._stats: dict[str, Any] = {
            "inferences": 0,
            "failures": 0,
            "fallbacks": 0,
            "total_infer_ms": 0.0,
            "last_infer_ms": 0.0,
            "peak_memory_mb": 0.0,
        }

    def _build_model(self, config: SeparationConfig, device: str) -> Any:
        raise NotImplementedError

    def load(self) -> None:

        self._loader.load()

    def close(self) -> None:
        self._loader.unload()

    @property
    def device(self) -> str:
        return self._loader.device

    @property
    def is_loaded(self) -> bool:
        return self._loader.is_loaded

    def get_stats(self) -> dict[str, Any]:
        stats = dict(self._stats)
        count = max(1, stats["inferences"])
        stats["avg_infer_ms"] = stats["total_infer_ms"] / count
        stats["device"] = self._loader.device
        stats["backend"] = self._config.name
        stats["precision"] = self._config.precision
        stats["sample_rate"] = self._config.sample_rate
        stats["load_time_ms"] = self._loader.load_time_ms
        stats["warmup_time_ms"] = self._loader.warmup_time_ms
        stats["last_error"] = self._last_error
        return stats

    def get_status(self) -> dict[str, Any]:

        with self._lock:
            inferences = self._stats["inferences"]
            failures = self._stats["failures"]
            fallbacks = self._stats["fallbacks"]
            fallback_active = self._fallback_active
            last_error = self._last_error
            total_ms = self._stats["total_infer_ms"]
            last_ms = self._stats["last_infer_ms"]
            peak_mem = self._stats["peak_memory_mb"]
        loaded = self._loader.is_loaded
        healthy = loaded and not fallback_active and failures == 0
        avg_ms = total_ms / inferences if inferences else 0.0
        return {
            "backend": self._config.name,
            "device": self._loader.device,
            "precision": self._config.precision,
            "sample_rate": self._config.sample_rate,
            "loaded": loaded,
            "healthy": healthy,

            "is_real_separation": loaded and not fallback_active,
            "fallback_active": fallback_active,
            "fallback_on_error": self._config.fallback_on_error,
            "inferences": inferences,
            "failures": failures,
            "fallbacks": fallbacks,
            "avg_infer_ms": avg_ms,
            "last_infer_ms": last_ms,
            "peak_memory_mb": peak_mem,
            "load_time_ms": self._loader.load_time_ms,
            "warmup_time_ms": self._loader.warmup_time_ms,
            "last_error": last_error,
        }

    def _warmup(self, model: Any, config: SeparationConfig, device: str) -> None:
        samples = config.chunk_size or 320
        chunk = silent_chunk(config.sample_rate, samples)
        ctx = AdapterContext(device=device, config=config)
        backend_input = self._adapter.to_backend(model, chunk, empty_target(), [], ctx)
        self._adapter.infer(model, backend_input, ctx)

    def separate(
        self,
        audio_chunk: AudioChunk,
        target: TargetSelection,
        all_tracks: list[SpeakerTrack],
    ) -> AudioChunk:
        start = time.perf_counter()
        try:
            model = self._loader.get()
            ctx = AdapterContext(device=self._loader.device, config=self._config)
            backend_input = self._adapter.to_backend(
                model, audio_chunk, target, all_tracks, ctx
            )
            output = run_with_timeout(
                lambda: self._adapter.infer(model, backend_input, ctx),
                self._timeout_s,
            )
            result = self._adapter.from_backend(output, audio_chunk, ctx)
            self._record_success(start)
            return result
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline
            return self._handle_failure(exc, audio_chunk)

    def _record_success(self, start: float) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        with self._lock:
            self._stats["inferences"] += 1
            self._stats["total_infer_ms"] += elapsed_ms
            self._stats["last_infer_ms"] = elapsed_ms
            self._fallback_active = False
            mem = memory_used_mb(self._loader.device)
            if mem > self._stats["peak_memory_mb"]:
                self._stats["peak_memory_mb"] = mem

    def _handle_failure(self, exc: Exception, audio_chunk: AudioChunk) -> AudioChunk:
        reason = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._stats["failures"] += 1
            self._last_error = reason
        if not self._config.fallback_on_error:
            raise exc
        with self._lock:
            self._stats["fallbacks"] += 1
            fallbacks = self._stats["fallbacks"]
            newly_degraded = not self._fallback_active
            self._fallback_active = True

        if newly_degraded:
            logger.error(
                "backend '%s' FAILED and is now DEGRADED to passthrough "
                "(returning UNSEPARATED audio): %s",
                self._config.name,
                reason,
            )
        elif fallbacks % self._warn_every == 0:
            logger.warning(
                "backend '%s' still degraded to passthrough after %d failed chunks: %s",
                self._config.name,
                fallbacks,
                reason,
            )
        return AudioChunk(
            timestamp_ms=audio_chunk.timestamp_ms,
            data=audio_chunk.data,
            sample_rate=audio_chunk.sample_rate,
            channels=audio_chunk.channels,
            metadata={
                **audio_chunk.metadata,
                "fallback": True,
                "backend": self._config.name,
                "fallback_reason": reason,
            },
        )
