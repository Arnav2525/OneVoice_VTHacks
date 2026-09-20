

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import Any

from bench._common import (
    default_targets,
    finalize_run,
    seed_everything,
    setup_run,
)
from bench._realtse import (
    SAMPLE_RATE,
    VARIANTS,
    eval_manifest,
    infer_separation,
    load_realtse,
    write_segment_wav,
)
from onevoice.telemetry import quality
from onevoice.telemetry.hardware import collect_hardware_info
from onevoice.telemetry.stats import summarize

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

def _align_triple(reference: Any, estimate: Any, mixture: Any) -> tuple[Any, Any, Any]:

    ref = np.asarray(reference, dtype=np.float64).flatten()
    est = np.asarray(estimate, dtype=np.float64).flatten()
    mix = np.asarray(mixture, dtype=np.float64).flatten()
    n = min(len(ref), len(est), len(mix))
    if n == 0:
        raise RuntimeError("empty audio after alignment")
    return ref[:n], est[:n], mix[:n]

def main() -> None:
    parser = argparse.ArgumentParser(
        description="REAL-TSE (causal BSRNN) Tier A offline separation quality benchmark"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data/dolphin_tier_a/tt",
        help="LRS2-format directory with mix/s1/s2.json (same corpus as Dolphin bench)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="spk_emb_causal_100",
        choices=VARIANTS,
        help="REAL-TSE checkpoint variant (causal vs offline, spk_emb vs tfmap_context)",
    )
    parser.add_argument(
        "--max-clips", type=int, default=None, help="cap number of target conditions evaluated"
    )
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="inference device (default: cpu — keeps local/CI safe)",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=2.0,
        help="evaluated mixture window length (matches Dolphin bench for latency parity)",
    )
    parser.add_argument(
        "--enroll-seconds",
        type=float,
        default=3.0,
        help="enrollment clip length, carved from the tail of the target's clean wav",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=2000.0,
        help="buffered-live window size used in E2E latency projection",
    )
    parser.add_argument(
        "--output-buffer-ms",
        type=float,
        default=100.0,
        help="assumed playback/output buffer in E2E latency projection",
    )
    args = parser.parse_args()

    if np is None:
        raise SystemExit("numpy is required: pip install -e '.[dolphin]'")

    test_dir = Path(args.test_dir)
    if not (test_dir / "mix.json").is_file():
        raise SystemExit(
            f"test data missing at {test_dir}. "
            "Run: python scripts/prepare_dolphin_tier_a_data.py"
        )

    seed_everything(args.seed)
    config: dict[str, Any] = {
        "benchmark": "realtse_offline",
        "variant": args.variant,
        "test_dir": str(test_dir),
        "sample_rate": SAMPLE_RATE,
        "seed": args.seed,
        "segment_seconds": args.segment_seconds,
        "enroll_seconds": args.enroll_seconds,
        "device": args.device,
        "eval_mode": "controlled_mix_disjoint_enroll",
    }
    targets = default_targets(config)
    run = setup_run("realtse_offline", config, args.output_dir)

    model, device, load_stats = load_realtse(args.variant, device=args.device)
    manifest = eval_manifest(
        test_dir, segment_seconds=args.segment_seconds, enroll_seconds=args.enroll_seconds
    )
    n_items = len(manifest)
    if args.max_clips is not None:
        n_items = min(n_items, args.max_clips)

    scratch = run.path / "segments"
    non_disjoint = sum(1 for m in manifest[:n_items] if not m["enroll_disjoint"])
    if non_disjoint:
        logging.getLogger("onevoice.bench").warning(
            "%d/%d clips too short for a disjoint enrollment window; "
            "enrollment overlaps the evaluated segment for those clips",
            non_disjoint,
            n_items,
        )

    per_clip: list[dict[str, float | str | None]] = []
    infer_ms_list: list[float] = []
    for idx in range(n_items):
        meta = manifest[idx]
        clip_tag = meta["clip_id"].replace("/", "_")

        mix_seg = write_segment_wav(
            meta["mix_path"],
            scratch / f"{clip_tag}_mix.wav",
            offset=0,
            length=meta["seg_len"],
        )
        ref_seg = write_segment_wav(
            meta["ref_path"],
            scratch / f"{clip_tag}_ref.wav",
            offset=0,
            length=meta["seg_len"],
        )
        enroll_seg = write_segment_wav(
            meta["ref_path"],
            scratch / f"{clip_tag}_enroll.wav",
            offset=meta["enroll_offset"],
            length=meta["enroll_len"],
        )

        import soundfile as sf

        mixture, _ = sf.read(str(mix_seg), dtype="float32")
        reference, _ = sf.read(str(ref_seg), dtype="float32")

        t0 = time.perf_counter()
        estimate = infer_separation(model, mix_seg, enroll_seg)
        infer_ms = (time.perf_counter() - t0) * 1000.0
        infer_ms_list.append(infer_ms)

        ref, est, mix = _align_triple(reference, estimate, mixture)
        metrics = quality.evaluate_pair(ref, est, mix, SAMPLE_RATE)
        row: dict[str, float | str | None] = {
            "clip_id": meta["clip_id"],
            "target": meta["target"],
            "ref_path": meta["ref_path"],
            "infer_ms": infer_ms,
            "enroll_disjoint": meta["enroll_disjoint"],
            **metrics,
        }
        per_clip.append(row)
        logging.getLogger("onevoice.bench").info(
            "%s SI-SNRi=%.2f dB infer=%.0f ms (SI-SNR=%.2f STOI=%s) enroll_disjoint=%s",
            meta["clip_id"],
            metrics["si_snr_improvement_db"],
            infer_ms,
            metrics["si_snr_db"],
            f"{metrics['stoi']:.2f}" if metrics["stoi"] is not None else "n/a",
            meta["enroll_disjoint"],
        )

    quality_summary: dict[str, object] = {}
    if per_clip:
        for key in per_clip[0]:
            if key in {"clip_id", "target", "ref_path", "enroll_disjoint"}:
                continue
            values = [float(v) for m in per_clip if (v := m[key]) is not None]
            quality_summary[key] = summarize(values) if values else None

    mean_snri = None
    median_snri = None
    gate_threshold_db = 7.0
    snri_values: list[float] = []
    snri_summary = quality_summary.get("si_snr_improvement_db")
    if isinstance(snri_summary, dict):
        mean_snri = snri_summary.get("avg")
        median_snri = snri_summary.get("median")
    for row in per_clip:
        v = row.get("si_snr_improvement_db")
        if v is not None:
            snri_values.append(float(v))
    gate_pass_count = sum(1 for v in snri_values if v >= gate_threshold_db)
    gate_total = len(snri_values)
    gate_min_total = 8
    gate_pass_fraction = 0.75
    gate_needed = math.ceil(gate_total * gate_pass_fraction)
    gate_ok = gate_total >= gate_min_total and gate_pass_count >= gate_needed

    infer_summary = summarize(infer_ms_list) if infer_ms_list else None
    mean_infer_ms = float(infer_summary["avg"]) if isinstance(infer_summary, dict) else None
    p95_infer_ms = float(infer_summary["p95"]) if isinstance(infer_summary, dict) else None
    buffered_e2e_ms = None
    if mean_infer_ms is not None:
        buffered_e2e_ms = (
            float(args.window_ms) + mean_infer_ms + float(args.output_buffer_ms)
        )

    aggregate: dict[str, object] = {
        "benchmark": "realtse_offline",
        "duration_s": 0.0,
        "quality": quality_summary,
        "realtse": {
            "variant": load_stats.variant,
            "causal": load_stats.causal,
            "device": load_stats.device,
            "param_count": load_stats.param_count,
            "clips_evaluated": len(per_clip),
            "mean_si_snr_improvement_db": mean_snri,
            "median_si_snr_improvement_db": median_snri,
            "gate_threshold_db": gate_threshold_db,
            "gate_pass_count": gate_pass_count,
            "gate_total": gate_total,
            "gate_passed": gate_ok,
            "infer_ms": infer_summary,
            "mean_infer_ms": mean_infer_ms,
            "p95_infer_ms": p95_infer_ms,
            "buffered_e2e_ms": buffered_e2e_ms,
            "window_ms": args.window_ms,
            "output_buffer_ms": args.output_buffer_ms,
            "eval_mode": "controlled_mix_disjoint_enroll",
            "non_disjoint_enroll_clips": non_disjoint,
        },
        "stability": {
            "clips": len(per_clip),
            "separator_is_real_separation": True,
        },
    }

    hardware = collect_hardware_info()
    report = finalize_run(run, aggregate, targets, per_result_rows=per_clip, hardware=hardware)
    print(report)
    if snri_values:
        print("\n--- REAL-TSE Tier A consistency gate (corpus) ---")
        for row in per_clip:
            snri = row.get("si_snr_improvement_db")
            if snri is None:
                continue
            mark = "PASS" if float(snri) >= gate_threshold_db else "FAIL"
            leak = "" if row["enroll_disjoint"] else " [enroll overlaps eval segment]"
            print(
                f"  {row['clip_id']}: SI-SNRi={float(snri):+.2f} dB "
                f"infer={float(row['infer_ms']):.0f} ms [{mark}]{leak}"
            )
        if median_snri is not None:
            print(f"\nMedian SI-SNRi: {median_snri:+.2f} dB")
        if mean_snri is not None:
            print(f"Mean SI-SNRi:   {mean_snri:+.2f} dB")
        print(
            f"Gate: {gate_pass_count}/{gate_total} targets >= {gate_threshold_db:.0f} dB "
            f"(need >={gate_needed}/{gate_total}, {gate_pass_fraction:.0%})"
        )

    if mean_infer_ms is not None and p95_infer_ms is not None:
        print("\n--- Latency (per %.1fs segment) ---" % args.segment_seconds)
        print(f"Mean infer: {mean_infer_ms:.0f} ms")
        print(f"P95 infer:  {p95_infer_ms:.0f} ms")
        if buffered_e2e_ms is not None:
            print(
                f"Projected buffered E2E: "
                f"{args.window_ms:.0f} (window) + {mean_infer_ms:.0f} (infer) + "
                f"{args.output_buffer_ms:.0f} (outbuf) = {buffered_e2e_ms:.0f} ms"
            )
            print(
                f"VERDICT: {args.variant} per-segment = {mean_infer_ms:.0f} ms -> "
                f"buffered-live ~{buffered_e2e_ms:.0f} ms; "
                f"{'VIABLE' if mean_infer_ms < float(args.window_ms) else 'NOT VIABLE'} "
                f"on an RTX Victus"
            )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
