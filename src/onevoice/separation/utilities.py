

from __future__ import annotations

import array
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, TypeVar

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.exceptions import (
    BackendInitializationError,
    InferenceTimeoutError,
)

T = TypeVar("T")

def import_torch() -> Any:

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise BackendInitializationError(
            "torch is required for this backend; install it or use "
            "backend.name=passthrough"
        ) from exc
    return torch

def import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise BackendInitializationError(
            "numpy is required for this backend"
        ) from exc
    return np

def select_device(preference: str) -> str:

    pref = preference.lower()
    if pref == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    cuda_ok = torch.cuda.is_available()
    if pref == "auto":
        return "cuda:0" if cuda_ok else "cpu"
    if pref.startswith("cuda"):
        return pref if cuda_ok else "cpu"
    return "cpu"

def resolve_dtype(precision: str, torch_module: Any) -> Any:
    return {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[precision]

def is_oom_error(exc: BaseException) -> bool:

    name = type(exc).__name__
    if name in {"OutOfMemoryError", "BackendOOMError"}:
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message

_TIMEOUT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="onevoice-sep")

def run_with_timeout(fn: Callable[[], T], timeout_s: float) -> T:

    future = _TIMEOUT_EXECUTOR.submit(fn)
    try:
        return future.result(timeout=timeout_s)
    except FuturesTimeoutError as exc:
        raise InferenceTimeoutError(
            f"inference exceeded {timeout_s * 1000:.0f} ms budget"
        ) from exc

def memory_used_mb(device: str) -> float:

    if device.startswith("cuda"):
        try:
            import torch

            return float(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:  # pragma: no cover - driver-dependent
            return 0.0
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:  # pragma: no cover
        return 0.0

def resample_waveform(wav: Any, src_rate: int, dst_rate: int) -> Any:

    if src_rate == dst_rate:
        return wav
    np = import_numpy()
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    n_in = arr.shape[0]
    if n_in == 0 or dst_rate <= 0 or src_rate <= 0:
        return arr
    n_out = max(1, int(round(n_in * dst_rate / src_rate)))
    positions = np.linspace(0.0, n_in - 1, num=n_out)
    return np.interp(positions, np.arange(n_in), arr).astype(np.float32)

def fit_length(wav: Any, length: int) -> Any:

    np = import_numpy()
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if length <= 0 or arr.shape[0] == length:
        return arr
    if arr.shape[0] > length:
        return arr[:length]
    pad = np.zeros(length - arr.shape[0], dtype=np.float32)
    return np.concatenate([arr, pad])

def chunk_to_float_list(data: Any) -> list[float]:

    if data is None:
        return []
    if isinstance(data, (list, array.array)):
        return [float(x) for x in data]
    if hasattr(data, "flatten"):
        return [float(x) for x in data.flatten().tolist()]
    return [float(x) for x in data]

def silent_chunk(sample_rate: int, samples: int, channels: int = 1) -> AudioChunk:

    buffer = array.array("f", [0.0] * (samples * channels))
    return AudioChunk(
        timestamp_ms=0.0,
        data=buffer,
        sample_rate=sample_rate,
        channels=channels,
        metadata={"warmup": True},
    )

def empty_target() -> TargetSelection:
    return TargetSelection(timestamp_ms=0.0, selected_speaker=None)

def make_output_chunk(
    samples: Any, reference: AudioChunk, extra_metadata: dict[str, Any] | None = None
) -> AudioChunk:

    metadata = dict(reference.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    return AudioChunk(
        timestamp_ms=reference.timestamp_ms,
        data=samples,
        sample_rate=reference.sample_rate,
        channels=reference.channels,
        metadata=metadata,
    )

class _ThreadSafeCounter:
    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, by: int = 1) -> int:
        with self._lock:
            self._value += by
            return self._value

    @property
    def value(self) -> int:
        return self._value
