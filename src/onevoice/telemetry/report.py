

from __future__ import annotations

from typing import Any

from onevoice.telemetry.thresholds import (
    CheckStatus,
    Evaluation,
    Sprint0Targets,
)

def evaluate_aggregate(
    aggregate: dict[str, Any], targets: Sprint0Targets
) -> Evaluation:

    ev = Evaluation()
    latency = aggregate.get("latency_ms", {})
    stability = aggregate.get("stability", {})
    resources = aggregate.get("resources", {})

    ev.real_separation = stability.get("separator_is_real_separation")

    e2e = latency.get("end_to_end", {})
    if e2e:
        p95 = e2e.get("p95", 0.0)
        ev.add(
            "end_to_end_latency",
            p95,
            targets.end_to_end_latency_ms_max,
            p95 <= targets.end_to_end_latency_ms_max,
            detail="P95 end-to-end latency",
        )

    rtf = aggregate.get("real_time_factor")
    if rtf is not None:
        ev.add(
            "real_time_factor",
            rtf,
            targets.real_time_factor_max,
            rtf < targets.real_time_factor_max,
            detail="separation time / chunk duration",
        )

    crashes = stability.get("crashes")
    if crashes is not None:
        ev.add(
            "no_crashes",
            float(crashes),
            float(targets.max_crashes),
            crashes <= targets.max_crashes,
        )

    overflow = _total_overflow(stability)
    if overflow is not None:
        ev.add(
            "no_queue_overflow",
            float(overflow),
            float(targets.max_queue_overflow),
            overflow <= targets.max_queue_overflow,
            detail="audio+video+playback drops",
        )

    fps = aggregate.get("throughput_fps")
    if fps is not None:
        ev.add(
            "throughput",
            fps,
            targets.min_throughput_fps,
            fps >= targets.min_throughput_fps,
            detail="processed results per second",
        )

    drift = stability.get("max_drift_ms")
    if drift is not None:
        ev.add(
            "av_drift",
            float(drift),
            targets.max_drift_ms,
            drift <= targets.max_drift_ms,
        )

    si_snri = _quality_value(aggregate, "si_snr_improvement_db")
    if si_snri is not None:
        ev.add(
            "audio_quality",
            si_snri,
            targets.si_snr_improvement_db_min,
            si_snri >= targets.si_snr_improvement_db_min,
            detail="mean SI-SNRi",
        )

    switch = aggregate.get("switch_latency_ms", {}).get("p95")
    if switch is not None:
        ev.add(
            "switch_latency",
            switch,
            targets.max_switch_latency_ms,
            switch <= targets.max_switch_latency_ms,
            detail="P95 target-switch gap",
        )

    _ = resources
    return ev

def _total_overflow(stability: dict[str, Any]) -> int | None:
    keys = ("audio_drops", "video_drops", "playback_drops")
    present = [stability[k] for k in keys if k in stability]
    if not present:
        return None
    return int(sum(present))

def _quality_value(aggregate: dict[str, Any], key: str) -> float | None:
    quality = aggregate.get("quality", {})
    value = quality.get(key)
    if isinstance(value, dict):
        avg = value.get("avg")
        return float(avg) if avg is not None else None
    if value is None:
        return None
    return float(value)

def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"

def _latency_table(latency: dict[str, Any]) -> list[str]:
    header = f"  {'stage':<18}{'avg':>9}{'p50':>9}{'p95':>9}{'p99':>9}{'max':>9}"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for stage, s in latency.items():
        lines.append(
            f"  {stage:<18}"
            f"{s.get('avg', 0.0):>9.2f}"
            f"{s.get('median', 0.0):>9.2f}"
            f"{s.get('p95', 0.0):>9.2f}"
            f"{s.get('p99', 0.0):>9.2f}"
            f"{s.get('max', 0.0):>9.2f}"
        )
    return lines

def _backend_info(aggregate: dict[str, Any]) -> tuple[Any, Any, Any]:
    stability = aggregate.get("stability", {})
    backend = stability.get("separator_backend")
    is_real = stability.get("separator_is_real_separation")
    fallback = stability.get("separator_fallback_active")
    return backend, is_real, fallback

def _passthrough_banner(backend: Any) -> list[str]:
    return [
        "",
        "!" * 64,
        f"!! WARNING: NO REAL SPEECH SEPARATION -- backend '{backend}' is "
        "passthrough/fallback.",
        "!! Metrics below reflect UNSEPARATED passthrough audio, not target "
        "separation.",
        "!" * 64,
    ]

def build_report(
    name: str,
    aggregate: dict[str, Any],
    targets: Sprint0Targets,
    hardware: dict[str, Any] | None = None,
) -> str:
    ev = evaluate_aggregate(aggregate, targets)
    backend, is_real, fallback = _backend_info(aggregate)
    degraded = backend is not None and (is_real is False or fallback is True)

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"OneVoice Benchmark Report — {name}")
    lines.append("=" * 64)

    if degraded:
        lines.extend(_passthrough_banner(backend))

    if backend is not None:
        lines.append("")
        lines.append("ACTIVE BACKEND")
        lines.append(f"  backend          : {backend}")
        lines.append(f"  real_separation  : {is_real}")
        if fallback is not None:
            lines.append(f"  fallback_active  : {fallback}")

    if hardware:
        lines.append("")
        lines.append("HARDWARE")
        lines.append(f"  platform : {hardware.get('platform')}")
        lines.append(f"  cpu      : {hardware.get('processor')} "
                     f"({hardware.get('cpu_count_logical')} logical)")
        lines.append(f"  ram      : {hardware.get('total_ram_mb')} MB")
        gpus = hardware.get("gpus") or []
        gpu_desc = ", ".join(g.get("name", "?") for g in gpus) if gpus else "none"
        lines.append(f"  gpu      : {gpu_desc}")

    lines.append("")
    lines.append("CONFIGURATION")
    lines.append(f"  duration_s        : {aggregate.get('duration_s')}")
    lines.append(f"  results_collected : "
                 f"{aggregate.get('stability', {}).get('results_collected')}")

    latency = aggregate.get("latency_ms", {})
    if latency:
        lines.append("")
        lines.append("LATENCY (ms)")
        lines.extend(_latency_table(latency))

    lines.append("")
    lines.append("THROUGHPUT / REAL-TIME")
    lines.append(f"  throughput_fps    : {_fmt(aggregate.get('throughput_fps'))}")
    lines.append(f"  real_time_factor  : {_fmt(aggregate.get('real_time_factor'))}")

    stability = aggregate.get("stability", {})
    if stability:
        lines.append("")
        lines.append("STABILITY")
        for key in sorted(stability):
            lines.append(f"  {key:<20}: {stability[key]}")

    resources = aggregate.get("resources", {})
    if resources:
        lines.append("")
        lines.append("RESOURCE USAGE")
        for key in sorted(resources):
            lines.append(f"  {key:<24}: {_fmt(resources[key])}")

    quality = aggregate.get("quality", {})
    if quality:
        lines.append("")
        lines.append("AUDIO QUALITY")
        for key in sorted(quality):
            val = quality[key]
            if isinstance(val, dict):
                lines.append(f"  {key:<24}: avg={_fmt(val.get('avg'))} "
                             f"p95={_fmt(val.get('p95'))}")
            else:
                lines.append(f"  {key:<24}: {_fmt(val)}")

    switch = aggregate.get("switch_latency_ms")
    if switch:
        lines.append("")
        lines.append("TARGET SWITCHING (ms)")
        lines.extend(_latency_table({"switch": switch}))

    lines.append("")
    lines.append("SPRINT 0 CHECKS")
    for check in ev.checks:
        mark = {
            CheckStatus.PASS: "[PASS]",
            CheckStatus.FAIL: "[FAIL]",
            CheckStatus.SKIP: "[SKIP]",
        }[check.status]
        target = "" if check.target is None else f" (target {_fmt(check.target)})"
        detail = f" — {check.detail}" if check.detail else ""
        measured = _fmt(check.measured)
        lines.append(f"  {mark} {check.name:<22} {measured}{target}{detail}")

    if degraded:
        lines.extend(_passthrough_banner(backend))

    lines.append("")
    lines.append("=" * 64)
    suffix = " (passthrough/fallback — NOT real separation)" if degraded else ""
    lines.append(f"RECOMMENDATION: {ev.recommendation().value}{suffix}")
    lines.append("=" * 64)
    return "\n".join(lines)
