

from __future__ import annotations

import array
import logging
import queue
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from onevoice.core.models.audio_chunk import AudioChunk

logger = logging.getLogger("onevoice.safety")

DEFAULT_MONITORED_CLASSES = frozenset(
    {"Siren", "Smoke detector, smoke alarm", "Baby cry, infant cry"}
)

@dataclass(frozen=True)
class SafetyEvent:

    class_name: str
    confidence: float

class SafetyClassifier(Protocol):

    def classify(self, audio: np.ndarray, sample_rate: int) -> SafetyEvent | None: ...

class FakeClassifier:

    def __init__(self, script: list[SafetyEvent | None] | None = None) -> None:
        self._script = list(script) if script else []
        self.calls: list[tuple[int, int]] = []
        self._done = threading.Event()

    def classify(self, audio: np.ndarray, sample_rate: int) -> SafetyEvent | None:
        self.calls.append((len(audio), sample_rate))
        self._done.set()
        if self._script:
            return self._script.pop(0)
        return None

    def wait_for_call(self, timeout: float = 2.0) -> bool:

        return self._done.wait(timeout)

    def reset(self) -> None:
        self._done.clear()

class YamnetClassifier:

    _MODEL_URL = "https://tfhub.dev/google/yamnet/1"

    def __init__(self, monitored_classes: set[str] | None = None) -> None:
        import csv

        import tensorflow_hub as hub

        self._model = hub.load(self._MODEL_URL)
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path, encoding="utf-8") as f:
            rows = list(csv.reader(f))[1:]
        self._class_names = [row[2] for row in rows]
        self._monitored_classes = monitored_classes

    def classify(self, audio: np.ndarray, sample_rate: int) -> SafetyEvent | None:
        if sample_rate != 16_000:
            raise ValueError(
                f"YamnetClassifier requires 16kHz audio (YAMNet's native "
                f"training rate, not resampled internally); got {sample_rate}"
            )
        waveform = np.asarray(audio, dtype=np.float32)
        scores, _embeddings, _spectrogram = self._model(waveform)

        mean_scores = np.mean(scores.numpy(), axis=0)

        if self._monitored_classes:
            candidates = [
                (name, float(mean_scores[i]))
                for i, name in enumerate(self._class_names)
                if name in self._monitored_classes
            ]
            if not candidates:
                return None
            name, confidence = max(candidates, key=lambda c: c[1])
        else:
            top_index = int(np.argmax(mean_scores))
            name = self._class_names[top_index]
            confidence = float(mean_scores[top_index])
        return SafetyEvent(name, confidence)

def _chunk_to_float32(chunk: AudioChunk) -> np.ndarray:

    data = chunk.data
    if isinstance(data, array.array):
        return np.frombuffer(data, dtype=np.float32)
    if hasattr(data, "astype"):
        return np.asarray(data, dtype=np.float32).reshape(-1)
    return np.asarray(data, dtype=np.float32).reshape(-1)

class SafetyMonitor:

    def __init__(
        self,
        classifier: SafetyClassifier,
        monitored_classes: set[str] | None = None,
        activate_thresh: float = 0.5,
        release_hold_s: float = 3.0,
        window_s: float = 0.96,
        hop_s: float = 0.48,
        sample_rate: int = 16_000,
        on_state_change: Callable[[bool, SafetyEvent | None], None] | None = None,
    ) -> None:
        self._classifier = classifier
        self._monitored_classes = monitored_classes
        self._activate_thresh = activate_thresh
        self._release_hold_s = release_hold_s
        self._window_samples = int(window_s * sample_rate)
        self._hop_samples = max(1, int(hop_s * sample_rate))
        self._sample_rate = sample_rate
        self._on_state_change = on_state_change

        self._lock = threading.Lock()
        self._buffer: list[np.ndarray] = []
        self._buffered_samples = 0
        self._samples_since_classify = 0
        self._active = False
        self._last_event_ts_ms: float | None = None
        self._latest_raw_chunk: AudioChunk | None = None

        self._queue: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        self._worker = threading.Thread(
            target=self._worker_loop, name="safety-monitor", daemon=True
        )
        self._worker.start()

    def stop(self) -> None:
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=10.0)
            self._worker = None

    def write(self, chunk: AudioChunk) -> None:
        samples = _chunk_to_float32(chunk)
        n = len(samples)
        if n == 0:
            return
        enqueue: tuple[np.ndarray, float] | None = None
        released = False
        with self._lock:
            self._latest_raw_chunk = chunk
            self._buffer.append(samples)
            self._buffered_samples += n
            self._samples_since_classify += n
            released = self._check_release_locked(chunk.timestamp_ms)

            ready = (
                self._buffered_samples >= self._window_samples
                and self._samples_since_classify >= self._hop_samples
            )
            if ready:
                window = np.concatenate(self._buffer)[-self._window_samples :]
                self._samples_since_classify = 0

                self._buffer = [window]
                self._buffered_samples = len(window)
                enqueue = (window, chunk.timestamp_ms)
        if released:
            self._fire_state_change(False, None)
        if enqueue is not None:
            self._queue.put(enqueue)

    def _check_release_locked(self, now_ms: float) -> bool:

        if self._active and self._last_event_ts_ms is not None:
            if (now_ms - self._last_event_ts_ms) / 1000.0 >= self._release_hold_s:
                self._active = False
                self._last_event_ts_ms = None
                return True
        return False

    def _fire_state_change(self, active: bool, event: SafetyEvent | None) -> None:
        if self._on_state_change is None:
            return
        try:
            self._on_state_change(active, event)
        except Exception:  # noqa: BLE001 - a bad callback must not kill the worker
            logger.exception(
                "on_state_change callback failed -- override state unaffected"
            )

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            window, ts_ms = item
            try:
                event = self._classifier.classify(window, self._sample_rate)
            except Exception:  # noqa: BLE001 - one bad window must not kill the worker
                logger.exception(
                    "safety classifier raised; treating this window as no event"
                )
                event = None

            qualifies = (
                event is not None
                and event.confidence >= self._activate_thresh
                and (
                    not self._monitored_classes
                    or event.class_name in self._monitored_classes
                )
            )
            activated_event: SafetyEvent | None = None
            released = False
            with self._lock:
                if qualifies:
                    was_active = self._active
                    self._active = True
                    self._last_event_ts_ms = ts_ms
                    if not was_active:
                        activated_event = event
                else:
                    released = self._check_release_locked(ts_ms)
            if activated_event is not None:
                self._fire_state_change(True, activated_event)
            elif released:
                self._fire_state_change(False, None)

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def latest_raw_chunk(self) -> AudioChunk | None:
        with self._lock:
            return self._latest_raw_chunk

class SafetyGatedSink:

    def __init__(self, real_sink: Any, monitor: SafetyMonitor) -> None:
        self._real_sink = real_sink
        self._monitor = monitor

    def start(self) -> None:
        self._real_sink.start()

    def stop(self) -> None:
        self._real_sink.stop()

    def write(self, chunk: AudioChunk) -> None:
        if self._monitor.is_active():
            override = self._monitor.latest_raw_chunk()
            self._real_sink.write(override if override is not None else chunk)
        else:
            self._real_sink.write(chunk)

DEFAULT_ACTIVATE_THRESH = 0.5
DEFAULT_RELEASE_HOLD_S = 3.0

YAMNET_SAMPLE_RATE = 16_000

def _yamnet_factory(monitored_classes: set[str]) -> SafetyClassifier:
    return YamnetClassifier(monitored_classes=monitored_classes)

def build_safety_monitor(
    config: dict[str, Any],
    sample_rate: int,
    on_state_change: Callable[[bool, SafetyEvent | None], None] | None = None,
    classifier_factory: Callable[[set[str]], SafetyClassifier] | None = None,
) -> SafetyMonitor | None:

    settings = config.get("safety") or {}
    if not settings.get("enabled", True):
        logger.info("Alarm passthrough is switched off in this config.")
        return None

    if sample_rate != YAMNET_SAMPLE_RATE:
        logger.warning(
            "ALARM PASSTHROUGH DISABLED: the detector needs %d Hz audio but this "
            "session runs at %d Hz. A fire alarm will NOT interrupt isolation.",
            YAMNET_SAMPLE_RATE,
            sample_rate,
        )
        return None

    requested = settings.get("monitored_classes")
    monitored = set(requested) if requested else set(DEFAULT_MONITORED_CLASSES)

    factory = classifier_factory or _yamnet_factory
    try:
        classifier = factory(monitored)
    except Exception:  # noqa: BLE001 - safety must never block a session from starting
        logger.warning(
            "ALARM PASSTHROUGH DISABLED: could not load the alarm detector "
            "(install the 'safety' extra: pip install -e '.[safety]'). The session "
            "will run normally, but a fire alarm will NOT interrupt isolation.",
            exc_info=True,
        )
        return None

    return SafetyMonitor(
        classifier,
        monitored_classes=monitored,
        activate_thresh=float(
            settings.get("activate_thresh", DEFAULT_ACTIVATE_THRESH)
        ),
        release_hold_s=float(settings.get("release_hold_s", DEFAULT_RELEASE_HOLD_S)),
        sample_rate=sample_rate,
        on_state_change=on_state_change,
    )
