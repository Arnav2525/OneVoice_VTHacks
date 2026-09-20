

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

from onevoice.audio.io import MockAudioSink, MockAudioSource
from onevoice.selection.first_track_selector import FirstTrackSelector
from onevoice.separation.registry import create_separator
from onevoice.streaming.pipeline import StreamingPipeline
from onevoice.telemetry.hardware import collect_hardware_info
from onevoice.telemetry.report import evaluate_aggregate
from onevoice.telemetry.results import RunDirectory
from onevoice.telemetry.thresholds import Sprint0Targets
from onevoice.video.capture import MockVideoSource
from onevoice.video.trackers.stub_tracker import StubFaceTracker

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger("onevoice.bench")

def build_argparser(
    description: str, default_duration: float = 30.0
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None, help="experiment YAML path")
    parser.add_argument(
        "--duration", type=float, default=default_duration, help="run seconds"
    )
    parser.add_argument("--output-dir", type=str, default="runs", help="results base")
    parser.add_argument(
        "--live", action="store_true", help="use real camera/microphone hardware"
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    return parser

def load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    if yaml is None:  # pragma: no cover
        raise RuntimeError("pyyaml required to load config")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass

def _audio_params(config: dict[str, Any]) -> tuple[int, int, int]:
    audio = config.get("audio", {})
    sample_rate = int(audio.get("sample_rate", 16000))
    channels = int(audio.get("channels", 1))
    chunk_samples = int(audio.get("chunk_samples", 320))
    return sample_rate, channels, chunk_samples

def make_separator(config: dict[str, Any]) -> Any:

    return create_separator(config.get("backend", {}))

def build_pipeline(
    config: dict[str, Any],
    on_result: Any,
    live: bool = False,
    selector: Any | None = None,
    separator: Any | None = None,
    face_tracker: Any | None = None,
) -> tuple[StreamingPipeline, float]:

    sample_rate, channels, chunk_samples = _audio_params(config)
    chunk_duration_ms = 1000.0 * chunk_samples / sample_rate

    if live:
        from onevoice.audio.io import MicrophoneSource, SpeakerSink
        from onevoice.video.capture import WebcamSource

        audio_source: Any = MicrophoneSource(sample_rate, channels, chunk_samples)
        audio_sink: Any = SpeakerSink(sample_rate, channels)
        video_source: Any = WebcamSource()
    else:
        audio_source = MockAudioSource(sample_rate, channels, chunk_samples)
        audio_sink = MockAudioSink()
        video_source = MockVideoSource()

    pipeline = StreamingPipeline(
        audio_source=audio_source,
        audio_sink=audio_sink,
        video_source=video_source,
        face_tracker=face_tracker or StubFaceTracker(),
        target_selector=selector or FirstTrackSelector(),
        target_separator=separator or make_separator(config),
        on_result=on_result,
    )
    return pipeline, chunk_duration_ms

def setup_run(name: str, config: dict[str, Any], output_dir: str) -> RunDirectory:
    run = RunDirectory(name, config=config, base_dir=output_dir)
    run.attach_logging()
    run.write_config()
    run.write_hardware(collect_hardware_info())
    logger.info("run %s started at %s", name, run.path)
    return run

def finalize_run(
    run: RunDirectory,
    aggregate: dict[str, Any],
    targets: Sprint0Targets,
    per_result_rows: list[dict[str, Any]] | None = None,
    hardware: dict[str, Any] | None = None,
) -> str:
    from onevoice.telemetry.report import build_report

    evaluation = evaluate_aggregate(aggregate, targets)
    metrics = dict(aggregate)
    metrics["evaluation"] = evaluation.to_dict()
    run.write_metrics(metrics)
    if per_result_rows:
        run.write_per_result_csv(per_result_rows)
    report = build_report(aggregate.get("benchmark", run.benchmark_name), aggregate,
                          targets, hardware=hardware)
    run.write_report(report)
    logger.info("run complete: %s", evaluation.recommendation().value)
    run.detach_logging()
    return report

def default_targets(config: dict[str, Any]) -> Sprint0Targets:
    overrides = config.get("targets", {})
    if not overrides:
        return Sprint0Targets()
    base = Sprint0Targets()
    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    fields.update({k: v for k, v in overrides.items() if k in fields})
    return Sprint0Targets(**fields)

def resolve_config_path(cli_path: str | None) -> str | None:
    if cli_path is not None:
        return cli_path
    default = Path("configs/experiments/smoke.yaml")
    return str(default) if default.exists() else None
