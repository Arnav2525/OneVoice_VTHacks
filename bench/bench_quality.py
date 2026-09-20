

from __future__ import annotations

import logging

from bench._common import (
    build_argparser,
    default_targets,
    finalize_run,
    load_config,
    make_separator,
    resolve_config_path,
    seed_everything,
    setup_run,
)
from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.telemetry import quality
from onevoice.telemetry.hardware import collect_hardware_info
from onevoice.telemetry.stats import summarize

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

def _tone(freq: float, duration_s: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    tone = 0.4 * np.sin(2 * np.pi * freq * t)
    return np.asarray(tone, dtype=np.float64)

def _run_chunked(separator, mixture, sample_rate, channels, chunk_samples):  # type: ignore[no-untyped-def]
    tracks = [
        SpeakerTrack(track_id="spk-0", bounding_box=(0, 0, 0.5, 1),
                     confidence=0.9, metadata={}),
        SpeakerTrack(track_id="spk-1", bounding_box=(0.5, 0, 0.5, 1),
                     confidence=0.9, metadata={}),
    ]
    target = TargetSelection(timestamp_ms=0.0, selected_speaker=tracks[0])
    out = []
    for start in range(0, len(mixture), chunk_samples):
        block = mixture[start:start + chunk_samples]
        chunk = AudioChunk(
            timestamp_ms=float(start),
            data=block.tolist(),
            sample_rate=sample_rate,
            channels=channels,
            metadata={},
        )
        result = separator.separate(chunk, target, tracks)
        out.append(np.asarray(result.data, dtype=np.float64))
    return np.concatenate(out) if out else np.zeros(0)

def main() -> None:
    parser = build_argparser("OneVoice separation quality benchmark")
    parser.add_argument("--trials", type=int, default=8, help="synthetic mixtures")
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    args = parser.parse_args()

    if np is None:
        raise SystemExit("numpy is required: pip install -e '.[bench]'")

    seed_everything(args.seed)
    config = load_config(resolve_config_path(args.config))
    targets = default_targets(config)
    sample_rate = int(config.get("audio", {}).get("sample_rate", 16000))
    channels = int(config.get("audio", {}).get("channels", 1))
    chunk_samples = int(config.get("audio", {}).get("chunk_samples", 320))

    run = setup_run("quality", config, args.output_dir)
    separator = make_separator(config)

    per_trial: list[dict[str, float | None]] = []
    for trial in range(args.trials):
        f_target = 180.0 + 40.0 * trial
        f_interf = 320.0 + 55.0 * trial
        reference = _tone(f_target, args.clip_seconds, sample_rate)
        interferer = _tone(f_interf, args.clip_seconds, sample_rate)
        mixture = reference + interferer
        estimate = _run_chunked(
            separator, mixture, sample_rate, channels, chunk_samples
        )
        metrics = quality.evaluate_pair(reference, estimate, mixture, sample_rate)
        per_trial.append(metrics)

    quality_summary: dict[str, object] = {}
    keys = per_trial[0].keys() if per_trial else []
    for key in keys:
        values = [
            float(v) for m in per_trial if (v := m[key]) is not None
        ]
        if values:
            quality_summary[key] = summarize(values)
        else:
            quality_summary[key] = None

    aggregate: dict[str, object] = {
        "benchmark": "quality",
        "duration_s": 0.0,
        "quality": quality_summary,
        "stability": {"trials": args.trials},
    }
    rows = [{k: v for k, v in m.items()} for m in per_trial]
    hardware = collect_hardware_info()
    report = finalize_run(run, aggregate, targets, per_result_rows=rows,
                          hardware=hardware)
    print(report)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
