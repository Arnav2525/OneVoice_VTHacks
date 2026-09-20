

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from demo.safety import FakeClassifier, SafetyEvent
from demo.safety_demo import build_safety_pipeline

def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()

def test_mock_mode_wires_raw_tap_into_monitor() -> None:

    classifier = FakeClassifier()
    pipeline, monitor = build_safety_pipeline(
        config={}, live=False, classifier=classifier
    )
    pipeline.start()
    try:
        assert _wait_until(lambda: monitor.latest_raw_chunk() is not None)
    finally:
        pipeline.stop()

def test_mock_mode_never_triggers_with_default_fake_classifier() -> None:

    pipeline, monitor = build_safety_pipeline(config={}, live=False)
    pipeline.start()
    try:
        time.sleep(0.5)
        assert monitor.is_active() is False
    finally:
        pipeline.stop()
    assert monitor.is_active() is False

def test_scripted_classifier_activates_override_during_a_real_run() -> None:

    classifier = FakeClassifier(script=[SafetyEvent("Siren", 0.9)])
    pipeline, monitor = build_safety_pipeline(
        config={},
        live=False,
        classifier=classifier,
        activate_thresh=0.5,
        release_hold_s=10.0,
    )
    pipeline.start()
    try:
        assert _wait_until(lambda: monitor.is_active(), timeout=5.0)
    finally:
        pipeline.stop()

def test_stop_shuts_down_cleanly_without_double_starting_the_monitor() -> None:

    pipeline, monitor = build_safety_pipeline(config={}, live=False)
    pipeline.start()
    pipeline.stop()
    assert monitor.is_active() is False

    pipeline.start()
    pipeline.stop()
