

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"

class Recommendation(str, Enum):
    GO = "GO"
    PIVOT = "PIVOT"
    FAIL = "FAIL"

    INCONCLUSIVE = "INCONCLUSIVE"

@dataclass(frozen=True)
class Sprint0Targets:

    end_to_end_latency_ms_max: float = 150.0
    real_time_factor_max: float = 1.0
    max_crashes: int = 0
    max_queue_overflow: int = 0
    min_throughput_fps: float = 10.0
    max_drift_ms: float = 100.0

    si_snr_improvement_db_min: float = 7.0
    max_switch_latency_ms: float = 500.0

@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    measured: float | None
    target: float | None
    detail: str = ""

@dataclass
class Evaluation:
    checks: list[CheckResult] = field(default_factory=list)

    real_separation: bool | None = None

    def add(
        self,
        name: str,
        measured: float | None,
        target: float | None,
        passed: bool | None,
        detail: str = "",
    ) -> None:
        if passed is None or measured is None:
            status = CheckStatus.SKIP
        else:
            status = CheckStatus.PASS if passed else CheckStatus.FAIL
        self.checks.append(
            CheckResult(
                name=name,
                status=status,
                measured=measured,
                target=target,
                detail=detail,
            )
        )

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.FAIL]

    @property
    def passes(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status is CheckStatus.PASS]

    def recommendation(self) -> Recommendation:

        hard_gates = {
            "end_to_end_latency",
            "no_crashes",
            "real_time_factor",
            "audio_quality",
        }
        failed = self.failures
        if any(c.name in hard_gates for c in failed):
            return Recommendation.FAIL
        if self.real_separation is False:
            return Recommendation.INCONCLUSIVE
        if failed:
            return Recommendation.PIVOT
        return Recommendation.GO

    def to_dict(self) -> dict[str, object]:
        return {
            "recommendation": self.recommendation().value,
            "real_separation": self.real_separation,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "measured": c.measured,
                    "target": c.target,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }
