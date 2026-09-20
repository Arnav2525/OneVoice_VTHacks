

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Protocol

from onevoice.core.models.pipeline_result import PipelineResult
from onevoice.telemetry.report import build_report
from onevoice.telemetry.stats import summarize
from onevoice.telemetry.thresholds import Sprint0Targets

logger = logging.getLogger(__name__)

STAGE_ORDER = (
    "audio_capture",
    "video_capture",
    "audio_queue",
    "video_queue",
    "sync_wait",
    "face_tracking",
    "target_selection",
    "separation",
    "playback",
    "end_to_end",
)

class RunnablePipeline(Protocol):

    def start(self) -> None: ...

    def stop(self) -> None: ...

    @property
    def is_running(self) -> bool: ...

    def get_stats(self) -> dict[str, Any]: ...

class MetricProfiler(Protocol):
    def start_profiling(self) -> None: ...

    def stop_profiling(self) -> None: ...

    def get_metrics(self) -> dict[str, float]: ...

class PipelineBenchmarkRunner:

    def __init__(
        self,
        name: str,
        pipeline: RunnablePipeline | None = None,
        duration_s: float = 30.0,
        chunk_duration_ms: float = 20.0,
        targets: Sprint0Targets | None = None,
        gpu_profiler: MetricProfiler | None = None,
        system_profiler: MetricProfiler | None = None,
        poll_interval_s: float = 0.5,
    ) -> None:
        self.name = name
        self._pipeline = pipeline
        self._duration_s = duration_s
        self._chunk_duration_ms = chunk_duration_ms
        self._targets = targets or Sprint0Targets()
        self._gpu = gpu_profiler
        self._sys = system_profiler
        self._poll_interval_s = poll_interval_s

        self._stage_samples: dict[str, list[float]] = defaultdict(list)
        self._per_result_rows: list[dict[str, Any]] = []
        self._collected = 0
        self._crashes = 0
        self._wall_start = 0.0
        self._wall_end = 0.0
        self._pipeline_stats: dict[str, Any] = {}

    def bind(
        self, pipeline: RunnablePipeline, chunk_duration_ms: float | None = None
    ) -> None:

        self._pipeline = pipeline
        if chunk_duration_ms is not None:
            self._chunk_duration_ms = chunk_duration_ms

    def collect_result(self, result: PipelineResult) -> None:
        self._collected += 1
        row: dict[str, Any] = {"timestamp_ms": result.timestamp_ms}
        for key, value in result.latency_metrics.items():
            if key.endswith("_last"):
                stage = key[: -len("_last")]
                self._stage_samples[stage].append(float(value))
                row[stage] = float(value)
        speaker = result.active_target.selected_speaker
        row["target_id"] = speaker.track_id if speaker is not None else ""
        self._per_result_rows.append(row)

    def run_benchmark(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("run_benchmark requires a bound pipeline")
        if self._gpu is not None:
            self._gpu.start_profiling()
        if self._sys is not None:
            self._sys.start_profiling()

        self._wall_start = time.monotonic()
        self._pipeline.start()
        deadline = self._wall_start + self._duration_s
        try:
            while time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                time.sleep(min(self._poll_interval_s, remaining))
                if not self._pipeline.is_running:
                    self._crashes += 1
                    logger.error("Pipeline reported not running during benchmark")
                    break
        finally:
            self._pipeline.stop()
            self._wall_end = time.monotonic()
            try:
                self._pipeline_stats = self._pipeline.get_stats()
            except Exception:  # pragma: no cover - defensive
                self._pipeline_stats = {}
            if self._sys is not None:
                self._sys.stop_profiling()
            if self._gpu is not None:
                self._gpu.stop_profiling()

    @property
    def per_result_rows(self) -> list[dict[str, Any]]:
        return self._per_result_rows

    def aggregate(self) -> dict[str, Any]:
        wall = max(self._wall_end - self._wall_start, 1e-9)
        latency: dict[str, dict[str, float]] = {}
        for stage in STAGE_ORDER:
            if self._stage_samples.get(stage):
                latency[stage] = summarize(self._stage_samples[stage])
        for stage, samples in self._stage_samples.items():
            if stage not in latency and samples:
                latency[stage] = summarize(samples)

        throughput_fps = self._collected / wall if wall > 0 else 0.0
        sep = latency.get("separation", {})
        rtf = (sep.get("avg", 0.0)) / self._chunk_duration_ms if sep else 0.0

        resources: dict[str, float] = {}
        if self._sys is not None:
            resources.update(self._sys.get_metrics())
        if self._gpu is not None:
            resources.update(self._gpu.get_metrics())

        stability = dict(self._pipeline_stats)
        stability.update(
            {
                "results_collected": self._collected,
                "wall_time_s": wall,
                "throughput_fps": throughput_fps,
                "crashes": self._crashes,
            }
        )

        return {
            "benchmark": self.name,
            "duration_s": self._duration_s,
            "latency_ms": latency,
            "throughput_fps": throughput_fps,
            "real_time_factor": rtf,
            "stability": stability,
            "resources": resources,
        }

    def generate_report(self) -> str:
        return build_report(self.name, self.aggregate(), self._targets)
