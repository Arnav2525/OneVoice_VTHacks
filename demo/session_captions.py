from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from demo.session_state import SessionState

logger = logging.getLogger(__name__)

MAX_TRANSCRIPT_LINES = 300


def _cuda_visible() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_caption_settings(config: dict[str, Any] | None) -> dict[str, str]:
    """large-v3 on cuda, small on cpu; explicit ``captions:`` config wins."""
    captions = (config or {}).get("captions") or {}
    device = str(captions.get("device", "auto")).lower()
    if device == "auto":
        device = "cuda" if _cuda_visible() else "cpu"
    default_model = "large-v3" if device == "cuda" else "small"
    return {
        "model_size": str(captions.get("model_size") or default_model),
        "device": device,
        "compute_type": str(captions.get("compute_type") or "int8"),
    }


_WARMUP_SAMPLES = 3200


def _validated_transcriber(settings: dict[str, str]) -> Any:
    """Load the model, then run one tiny transcribe to prove it works.

    A GPU can be visible and the model can load while inference still fails
    (missing cuBLAS DLLs), so a cuda failure here falls back to cpu/small.
    """
    import numpy as np

    from demo.transcription import FasterWhisperTranscriber

    def _build(device: str, model_size: str) -> Any:
        transcriber = FasterWhisperTranscriber(
            model_size=model_size,
            device=device,
            compute_type=settings["compute_type"],
        )
        transcriber.transcribe(np.zeros(_WARMUP_SAMPLES, dtype=np.float32), 16_000)
        return transcriber

    try:
        return _build(settings["device"], settings["model_size"])
    except Exception as exc:
        if settings["device"] != "cuda":
            raise
        fallback = (
            "small" if settings["model_size"] == "large-v3" else settings["model_size"]
        )
        logger.warning(
            "Captions: CUDA unusable (%s: %s); falling back to cpu/%s",
            type(exc).__name__,
            exc,
            fallback,
        )
        return _build("cpu", fallback)


def build_transcriber_factory(config: dict[str, Any] | None) -> Callable[[], Any]:
    settings = resolve_caption_settings(config)
    logger.info("Captions ASR requested: %s", settings)
    return lambda: _validated_transcriber(settings)


def _default_transcriber() -> Any:
    return build_transcriber_factory(None)()


class SessionCaptions:
    def __init__(
        self,
        state: SessionState,
        transcriber_factory: Callable[[], Any] = _default_transcriber,
        sample_rate: int = 16_000,
    ) -> None:
        self._state = state
        self._transcriber_factory = transcriber_factory
        self._sample_rate = sample_rate
        self._lock = threading.Lock()
        self._enabled = False
        self._status = "off"
        self._error: str | None = None
        self._transcriber: Any = None
        self._sink: Any = None
        self._sink_epoch: int | None = None
        self._current: dict[str, Any] | None = None
        self._lines: deque[tuple[str, str]] = deque(maxlen=MAX_TRANSCRIPT_LINES)
        self._load_thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def toggle(self) -> None:
        with self._lock:
            turning_on = not self._enabled
            self._enabled = turning_on
            if not turning_on:
                self._retire_sink_locked()
                self._status = "ready" if self._transcriber is not None else "off"
        if turning_on:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._transcriber is not None:
                self._status = "ready"
                return
            if self._status == "loading":
                return
            self._status, self._error = "loading", None
            thread = threading.Thread(
                target=self._load, name="onevoice-captions-load", daemon=True
            )
            self._load_thread = thread
        thread.start()

    def _load(self) -> None:
        try:
            transcriber = self._transcriber_factory()
        except Exception as exc:
            logger.exception("Could not load the captions model")
            with self._lock:
                self._status = "error"
                self._error = str(exc) or type(exc).__name__
                self._enabled = False
                self._load_thread = None
            return
        with self._lock:
            self._transcriber = transcriber
            self._status = "ready" if self._enabled else "off"
            self._load_thread = None

    def on_output(self, chunk: Any, allowed: bool) -> None:

        with self._lock:
            if not self._enabled or self._transcriber is None:
                return
        requested, epoch = self._state.selection()
        with self._lock:
            if not self._enabled or self._transcriber is None:
                return
            if self._sink is not None and self._sink_epoch != epoch:
                self._retire_sink_locked()
            if not allowed or requested is None:
                return
            track_id = chunk.metadata.get("target_track_id")
            if track_id != requested:
                return
            if self._sink is None:
                self._start_sink_locked(epoch)
            sink = self._sink
        sink.write(chunk)

    def retire_current_sink(self) -> None:

        with self._lock:
            self._retire_sink_locked()

    def _start_sink_locked(self, epoch: int) -> None:
        from demo.transcription import TranscribingSink

        sink = TranscribingSink(
            str(epoch),
            self._transcriber,
            self._on_transcript,
            sample_rate=self._sample_rate,
        )
        sink.start()
        self._sink = sink
        self._sink_epoch = epoch

    def _retire_sink_locked(self) -> None:

        old = self._sink
        self._sink, self._sink_epoch, self._current = None, None, None
        if old is not None:
            threading.Thread(
                target=old.stop, name="onevoice-caption-sink-stop", daemon=True
            ).start()

    def _on_transcript(self, label: str, text: str, timestamp_ms: float) -> None:
        try:
            segment_epoch = int(label)
        except ValueError:
            return
        requested, current_epoch = self._state.selection()
        if requested is None or segment_epoch != current_epoch:
            return
        speaker = self._state.person_name(requested)
        with self._lock:
            self._current = {
                "track_id": requested,
                "text": text,
                "timestamp_ms": timestamp_ms,
            }
            if text.strip():
                self._lines.append((speaker, text.strip()))

    def transcript_text(self, max_chars: int = 12_000) -> str:
        with self._lock:
            lines = [f"{speaker}: {text}" for speaker, text in self._lines]
        kept: list[str] = []
        total = 0
        for line in reversed(lines):
            total += len(line) + 1
            if total > max_chars:
                break
            kept.append(line)
        return "\n".join(reversed(kept))

    def clear_transcript(self) -> None:
        with self._lock:
            self._lines.clear()

    def snapshot(self) -> dict[str, Any]:
        requested, _ = self._state.selection()
        with self._lock:
            current = self._current
            if current is not None and current["track_id"] != requested:
                current = None
            return {
                "enabled": self._enabled,
                "status": self._status,
                "error": self._error,
                "current": current,
            }

    def close(self) -> None:

        with self._lock:
            self._enabled = False
            self._lines.clear()
            self._retire_sink_locked()
            transcriber = self._transcriber
            self._transcriber = None
            self._status = "off"
            self._error = None
        del transcriber
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
