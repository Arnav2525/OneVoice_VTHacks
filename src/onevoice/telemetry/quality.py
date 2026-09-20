

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is required for quality metrics
    np = None  # type: ignore[assignment]

try:
    from pystoi import stoi as _stoi
except ImportError:  # pragma: no cover - optional dependency
    _stoi = None

try:
    from pesq import pesq as _pesq
except ImportError:  # pragma: no cover - optional dependency
    _pesq = None

_EPS = 1e-8

def _as_array(signal: Any) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required for quality metrics; pip install numpy")
    arr = np.asarray(signal, dtype=np.float64).flatten()
    return arr

def _align(reference: np.ndarray, estimate: np.ndarray) -> tuple[Any, Any]:
    length = min(len(reference), len(estimate))
    return reference[:length], estimate[:length]

def si_snr(reference: Any, estimate: Any) -> float:

    ref = _as_array(reference)
    est = _as_array(estimate)
    ref, est = _align(ref, est)
    if len(ref) == 0:
        return 0.0
    ref = ref - ref.mean()
    est = est - est.mean()
    ref_energy = float(np.dot(ref, ref)) + _EPS
    proj = (float(np.dot(est, ref)) / ref_energy) * ref
    noise = est - proj
    ratio = (float(np.dot(proj, proj)) + _EPS) / (float(np.dot(noise, noise)) + _EPS)
    return float(10.0 * np.log10(ratio))

def si_snr_improvement(reference: Any, estimate: Any, mixture: Any) -> float:

    return si_snr(reference, estimate) - si_snr(reference, mixture)

def signal_energy(signal: Any) -> float:

    arr = _as_array(signal)
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr**2))

def rms_dbfs(signal: Any) -> float:

    energy = signal_energy(signal)
    if energy <= 0:
        return float("-inf")
    return float(20.0 * np.log10((energy**0.5) + _EPS))

def clipping_ratio(signal: Any, threshold: float = 0.999) -> float:

    arr = _as_array(signal)
    if len(arr) == 0:
        return 0.0
    clipped = int(np.sum(np.abs(arr) >= threshold))
    return clipped / len(arr)

def stoi(reference: Any, estimate: Any, sample_rate: int) -> float | None:

    if _stoi is None:
        return None
    ref = _as_array(reference)
    est = _as_array(estimate)
    ref, est = _align(ref, est)
    return float(_stoi(ref, est, sample_rate, extended=False))

def pesq(
    reference: Any, estimate: Any, sample_rate: int, mode: str = "wb"
) -> float | None:

    if _pesq is None:
        return None
    if sample_rate not in (8000, 16000):
        return None
    ref = _as_array(reference)
    est = _as_array(estimate)
    ref, est = _align(ref, est)
    try:
        return float(_pesq(sample_rate, ref, est, mode))
    except Exception:  # pragma: no cover - pesq raises on degenerate input
        return None

def evaluate_pair(
    reference: Any,
    estimate: Any,
    mixture: Any,
    sample_rate: int,
) -> dict[str, float | None]:

    return {
        "si_snr_db": si_snr(reference, estimate),
        "si_snr_improvement_db": si_snr_improvement(reference, estimate, mixture),
        "estimate_energy": signal_energy(estimate),
        "estimate_rms_dbfs": rms_dbfs(estimate),
        "clipping_ratio": clipping_ratio(estimate),
        "stoi": stoi(reference, estimate, sample_rate),
        "pesq": pesq(reference, estimate, sample_rate),
    }
