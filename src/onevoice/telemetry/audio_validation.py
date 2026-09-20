

from __future__ import annotations

import array
import math
import random
from dataclasses import dataclass, field
from typing import Any

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection

CLIP_THRESHOLD = 0.999
MAX_CLIP_RATIO = 0.01
MAX_STABLE_PEAK = 10.0

def _tone(
    freq: float, sample_rate: int, n: int, amp: float, phase: float = 0.0
) -> list[float]:
    step = 2.0 * math.pi * freq / sample_rate
    return [amp * math.sin(step * i + phase) for i in range(n)]

def _mix(*signals: list[float]) -> array.array:
    n = max((len(s) for s in signals), default=0)
    out = array.array("f", [0.0] * n)
    for sig in signals:
        for i, v in enumerate(sig):
            out[i] += v
    return out

def _speaker(track_id: str, source_index: int) -> SpeakerTrack:
    return SpeakerTrack(
        track_id=track_id,
        bounding_box=(0.0, 0.0, 1.0, 1.0),
        confidence=1.0,
        metadata={"source_index": source_index},
    )

def _chunk(data: array.array, sample_rate: int, channels: int) -> AudioChunk:
    return AudioChunk(
        timestamp_ms=0.0,
        data=data,
        sample_rate=sample_rate,
        channels=channels,
        metadata={"validation": True},
    )

@dataclass(frozen=True)
class Scenario:
    name: str
    chunk: AudioChunk
    target: TargetSelection
    tracks: list[SpeakerTrack]

def build_scenarios(
    sample_rate: int = 16000, chunk_samples: int = 16000, seed: int = 1234
) -> list[Scenario]:

    rng = random.Random(seed)
    n = chunk_samples

    single = _mix(_tone(220.0, sample_rate, n, 0.4))
    two = _mix(
        _tone(220.0, sample_rate, n, 0.35),
        _tone(440.0, sample_rate, n, 0.35),
    )
    multi = _mix(
        _tone(180.0, sample_rate, n, 0.25),
        _tone(300.0, sample_rate, n, 0.25),
        _tone(520.0, sample_rate, n, 0.25),
    )
    noise = _mix(
        _tone(330.0, sample_rate, n, 0.3),
        [0.15 * (rng.random() * 2.0 - 1.0) for _ in range(n)],
    )

    spk0 = _speaker("spk0", 0)
    spk1 = _speaker("spk1", 1)

    def sel(track: SpeakerTrack | None) -> TargetSelection:
        return TargetSelection(timestamp_ms=0.0, selected_speaker=track)

    return [
        Scenario("single_speaker", _chunk(single, sample_rate, 1), sel(spk0), [spk0]),
        Scenario(
            "two_speakers", _chunk(two, sample_rate, 1), sel(spk0), [spk0, spk1]
        ),
        Scenario(
            "multi_speaker",
            _chunk(multi, sample_rate, 1),
            sel(spk0),
            [spk0, spk1, _speaker("spk2", 2)],
        ),
        Scenario(
            "background_noise", _chunk(noise, sample_rate, 1), sel(spk0), [spk0]
        ),
    ]

@dataclass
class ScenarioResult:
    name: str
    ok: bool
    sample_rate_ok: bool
    channels_ok: bool
    length_ok: bool
    finite_ok: bool
    clipping_ratio: float
    peak: float
    rms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "sample_rate_ok": self.sample_rate_ok,
            "channels_ok": self.channels_ok,
            "length_ok": self.length_ok,
            "finite_ok": self.finite_ok,
            "clipping_ratio": self.clipping_ratio,
            "peak": self.peak,
            "rms": self.rms,
            "error": self.error,
        }

def validate_output(
    input_chunk: AudioChunk, output_chunk: AudioChunk
) -> ScenarioResult:

    samples = list(output_chunk.data) if output_chunk.data is not None else []
    n = len(samples)
    finite_ok = all(math.isfinite(x) for x in samples)
    peak = max((abs(x) for x in samples), default=0.0)
    rms = math.sqrt(sum(x * x for x in samples) / n) if n else 0.0
    clipped = sum(1 for x in samples if abs(x) > CLIP_THRESHOLD)
    clipping_ratio = clipped / n if n else 0.0

    sample_rate_ok = output_chunk.sample_rate == input_chunk.sample_rate
    channels_ok = output_chunk.channels == input_chunk.channels
    length_ok = n == len(input_chunk.data)
    stable = finite_ok and peak <= MAX_STABLE_PEAK
    ok = (
        sample_rate_ok
        and channels_ok
        and length_ok
        and stable
        and clipping_ratio <= MAX_CLIP_RATIO
    )
    return ScenarioResult(
        name="",
        ok=ok,
        sample_rate_ok=sample_rate_ok,
        channels_ok=channels_ok,
        length_ok=length_ok,
        finite_ok=finite_ok,
        clipping_ratio=clipping_ratio,
        peak=peak,
        rms=rms,
    )

@dataclass
class ValidationReport:
    passed: bool
    scenarios: list[ScenarioResult] = field(default_factory=list)
    backend_status: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "scenarios": [s.to_dict() for s in self.scenarios],
            "backend_status": self.backend_status,
        }

def run_validation(
    separator: Any,
    sample_rate: int = 16000,
    chunk_samples: int = 16000,
    seed: int = 1234,
) -> ValidationReport:

    results: list[ScenarioResult] = []
    for scenario in build_scenarios(sample_rate, chunk_samples, seed):
        try:
            out = separator.separate(scenario.chunk, scenario.target, scenario.tracks)
            result = validate_output(scenario.chunk, out)
            result.name = scenario.name
        except Exception as exc:  # noqa: BLE001 - report, never raise
            result = ScenarioResult(
                name=scenario.name,
                ok=False,
                sample_rate_ok=False,
                channels_ok=False,
                length_ok=False,
                finite_ok=False,
                clipping_ratio=0.0,
                peak=0.0,
                rms=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)

    status = None
    if hasattr(separator, "get_status"):
        try:
            status = separator.get_status()
        except Exception:  # pragma: no cover - defensive
            status = None

    passed = all(r.ok for r in results)
    return ValidationReport(passed=passed, scenarios=results, backend_status=status)
