

from __future__ import annotations

import logging

from bench._common import (
    build_argparser,
    default_targets,
    finalize_run,
    load_config,
    resolve_config_path,
    seed_everything,
    setup_run,
)
from onevoice.telemetry.hardware import collect_hardware_info
from onevoice.telemetry.stats import summarize

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

def _make_click_train(
    n_clicks: int, spacing_samples: int, total_samples: int
) -> tuple[np.ndarray, list[int]]:
    signal = np.zeros(total_samples, dtype=np.float64)
    positions = []
    for i in range(n_clicks):
        pos = (i + 1) * spacing_samples
        if pos < total_samples:
            signal[pos] = 1.0
            positions.append(pos)
    return signal, positions

def _simulate_loopback(
    signal: np.ndarray, delay_samples: int, noise_level: float
) -> np.ndarray:
    out = np.zeros_like(signal)
    if delay_samples < len(signal):
        out[delay_samples:] = signal[: len(signal) - delay_samples]
    out += noise_level * np.random.randn(len(signal))
    return out

def _measure_latencies(
    output: np.ndarray,
    positions: list[int],
    sample_rate: int,
    search_samples: int,
) -> list[float]:
    latencies_ms = []
    for pos in positions:
        window = output[pos:pos + search_samples]
        if len(window) == 0:
            continue
        peak = int(np.argmax(np.abs(window)))
        latencies_ms.append(1000.0 * peak / sample_rate)
    return latencies_ms

def main() -> None:
    parser = build_argparser("OneVoice loopback latency benchmark")
    parser.add_argument("--clicks", type=int, default=20)
    parser.add_argument(
        "--inject-delay-ms", type=float, default=80.0,
        help="synthetic ground-truth delay (ignored with --live)"
    )
    args = parser.parse_args()

    if np is None:
        raise SystemExit("numpy is required: pip install -e '.[bench]'")

    seed_everything(args.seed)
    config = load_config(resolve_config_path(args.config))
    targets = default_targets(config)
    sample_rate = int(config.get("audio", {}).get("sample_rate", 16000))

    run = setup_run("loopback", config, args.output_dir)

    spacing = int(0.25 * sample_rate)
    total = spacing * (args.clicks + 2)
    signal, positions = _make_click_train(args.clicks, spacing, total)

    if args.live:  # pragma: no cover - requires audio hardware
        raise SystemExit(
            "Live loopback requires speaker+mic capture; run on hardware. "
            "Synthetic mode validates the cross-correlation estimator."
        )

    delay_samples = int(args.inject_delay_ms / 1000.0 * sample_rate)
    output = _simulate_loopback(signal, delay_samples, noise_level=0.001)
    latencies = _measure_latencies(output, positions, sample_rate, spacing)

    summary = summarize(latencies)
    aggregate: dict[str, object] = {
        "benchmark": "loopback",
        "duration_s": total / sample_rate,
        "latency_ms": {"end_to_end": summary},
        "stability": {
            "clicks_emitted": len(positions),
            "clicks_detected": len(latencies),
            "ground_truth_delay_ms": args.inject_delay_ms,
        },
    }
    rows = [{"click": i, "latency_ms": v} for i, v in enumerate(latencies)]
    hardware = collect_hardware_info()
    report = finalize_run(run, aggregate, targets, per_result_rows=rows,
                          hardware=hardware)
    print(report)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
