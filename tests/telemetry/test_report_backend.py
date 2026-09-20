

from onevoice.telemetry.report import build_report
from onevoice.telemetry.thresholds import Sprint0Targets

def _aggregate(**separator_stats):
    stability = {"results_collected": 10, "crashes": 0, **separator_stats}
    return {
        "benchmark": "unit",
        "duration_s": 1.0,
        "latency_ms": {},
        "throughput_fps": 10.0,
        "real_time_factor": 0.0,
        "stability": stability,
        "resources": {},
    }

def test_report_flags_passthrough_backend():
    agg = _aggregate(
        separator_backend="passthrough",
        separator_is_real_separation=False,
    )
    report = build_report("unit", agg, Sprint0Targets())
    assert "NO REAL SPEECH SEPARATION" in report
    assert "ACTIVE BACKEND" in report
    assert "passthrough" in report
    assert "NOT real separation" in report

def test_report_flags_fallback_backend():
    agg = _aggregate(
        separator_backend="speechbrain",
        separator_is_real_separation=False,
        separator_fallback_active=True,
    )
    report = build_report("unit", agg, Sprint0Targets())
    assert "NO REAL SPEECH SEPARATION" in report
    assert "fallback_active  : True" in report

def test_report_clean_for_real_backend():
    agg = _aggregate(
        separator_backend="speechbrain",
        separator_is_real_separation=True,
        separator_fallback_active=False,
    )
    report = build_report("unit", agg, Sprint0Targets())
    assert "NO REAL SPEECH SEPARATION" not in report
    assert "ACTIVE BACKEND" in report
    assert "real_separation  : True" in report
