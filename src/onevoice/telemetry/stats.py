

from __future__ import annotations

import math
from collections.abc import Sequence

STAT_KEYS = ("count", "avg", "median", "p95", "p99", "max", "min")

def percentile(values: Sequence[float], pct: float) -> float:

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    frac = rank - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac

def summarize(values: Sequence[float]) -> dict[str, float]:

    if not values:
        return {key: 0.0 for key in STAT_KEYS}
    ordered = sorted(values)
    total = float(sum(ordered))
    return {
        "count": float(len(ordered)),
        "avg": total / len(ordered),
        "median": percentile(ordered, 50.0),
        "p95": percentile(ordered, 95.0),
        "p99": percentile(ordered, 99.0),
        "max": ordered[-1],
        "min": ordered[0],
    }

def flatten_summary(prefix: str, summary: dict[str, float]) -> dict[str, float]:

    return {f"{prefix}_{key}": value for key, value in summary.items()}
