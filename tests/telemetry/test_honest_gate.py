

from onevoice.telemetry.report import evaluate_aggregate
from onevoice.telemetry.thresholds import (
    Evaluation,
    Recommendation,
    Sprint0Targets,
)

def test_default_si_snr_gate_is_seven_db():
    assert Sprint0Targets().si_snr_improvement_db_min == 7.0

def _aggregate(real_separation, si_snri=None, e2e_p95=100.0, crashes=0):
    stability = {"crashes": crashes, "separator_is_real_separation": real_separation}
    agg = {
        "latency_ms": {"end_to_end": {"p95": e2e_p95}},
        "real_time_factor": 0.5,
        "throughput_fps": 20.0,
        "stability": stability,
    }
    if si_snri is not None:
        agg["quality"] = {"si_snr_improvement_db": si_snri}
    return agg

def test_passthrough_all_green_is_inconclusive_not_go():
    ev = evaluate_aggregate(_aggregate(real_separation=False), Sprint0Targets())
    assert ev.recommendation() is Recommendation.INCONCLUSIVE

def test_real_backend_all_green_is_go():
    ev = evaluate_aggregate(
        _aggregate(real_separation=True, si_snri=8.0), Sprint0Targets()
    )
    assert ev.recommendation() is Recommendation.GO

def test_si_snri_below_seven_is_hard_fail_even_if_real():
    ev = evaluate_aggregate(
        _aggregate(real_separation=True, si_snri=5.0), Sprint0Targets()
    )
    assert ev.recommendation() is Recommendation.FAIL

def test_unknown_real_separation_is_backward_compatible_go():
    ev = evaluate_aggregate(_aggregate(real_separation=None), Sprint0Targets())
    assert ev.recommendation() is Recommendation.GO

def test_hard_fail_takes_precedence_over_inconclusive():

    ev = evaluate_aggregate(
        _aggregate(real_separation=False, crashes=3), Sprint0Targets()
    )
    assert ev.recommendation() is Recommendation.FAIL

def test_evaluation_real_separation_field_drives_verdict():
    ev = Evaluation(real_separation=False)
    ev.add("throughput", 20.0, 10.0, True)
    assert ev.recommendation() is Recommendation.INCONCLUSIVE
    ev2 = Evaluation(real_separation=True)
    ev2.add("throughput", 20.0, 10.0, True)
    assert ev2.recommendation() is Recommendation.GO
