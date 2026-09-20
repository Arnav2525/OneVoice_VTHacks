

from __future__ import annotations

import logging
import threading
import time

from bench._common import (
    build_argparser,
    build_pipeline,
    default_targets,
    finalize_run,
    load_config,
    resolve_config_path,
    seed_everything,
    setup_run,
)
from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.telemetry.benchmark_runner import PipelineBenchmarkRunner
from onevoice.telemetry.hardware import collect_hardware_info
from onevoice.telemetry.stats import summarize

def _now_ms() -> float:
    return time.monotonic() * 1000.0

class MultiFaceTracker:

    def __init__(self, track_ids: list[str]) -> None:
        self._track_ids = track_ids

    def process_frame(self, frame: Frame) -> list[SpeakerTrack]:
        n = len(self._track_ids)
        return [
            SpeakerTrack(
                track_id=tid,
                bounding_box=(i / n, 0.0, 1.0 / n, 1.0),
                confidence=0.9,
                metadata={"stub": True},
            )
            for i, tid in enumerate(self._track_ids)
        ]

class ScriptedSelector:

    def __init__(self, initial_id: str) -> None:
        self._current_id = initial_id
        self._lock = threading.Lock()

    def set_target(self, track_id: str) -> None:
        with self._lock:
            self._current_id = track_id

    def select_target(
        self, frame: Frame, tracks: list[SpeakerTrack]
    ) -> TargetSelection:
        with self._lock:
            current = self._current_id
        chosen = next((t for t in tracks if t.track_id == current), None)
        return TargetSelection(
            timestamp_ms=frame.timestamp_ms, selected_speaker=chosen
        )

class SwitchTimer:

    def __init__(
        self, selector: ScriptedSelector, track_ids: list[str], interval_s: float
    ) -> None:
        self._selector = selector
        self._track_ids = track_ids
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._pending_id: str | None = None
        self._request_ms = 0.0
        self._gaps_ms: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._idx = 0

    @property
    def gaps_ms(self) -> list[float]:
        return self._gaps_ms

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._idx = (self._idx + 1) % len(self._track_ids)
            new_id = self._track_ids[self._idx]
            with self._lock:
                self._pending_id = new_id
                self._request_ms = _now_ms()
            self._selector.set_target(new_id)

    def observe(self, target: TargetSelection) -> None:
        speaker = target.selected_speaker
        if speaker is None:
            return
        with self._lock:
            if self._pending_id is not None and speaker.track_id == self._pending_id:
                self._gaps_ms.append(_now_ms() - self._request_ms)
                self._pending_id = None

def main() -> None:
    parser = build_argparser("OneVoice target-switch benchmark", default_duration=30.0)
    parser.add_argument(
        "--switch-interval", type=float, default=1.0, help="seconds between switches"
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    config = load_config(resolve_config_path(args.config))
    targets = default_targets(config)

    track_ids = ["spk-0", "spk-1"]
    selector = ScriptedSelector(track_ids[0])
    switch_timer = SwitchTimer(selector, track_ids, args.switch_interval)

    run = setup_run("switch", config, args.output_dir)
    runner = PipelineBenchmarkRunner(
        name="switch", duration_s=args.duration, targets=targets
    )

    def on_result(result):  # type: ignore[no-untyped-def]
        runner.collect_result(result)
        switch_timer.observe(result.active_target)

    pipeline, chunk_ms = build_pipeline(
        config,
        on_result=on_result,
        live=args.live,
        selector=selector,
        face_tracker=MultiFaceTracker(track_ids),
    )
    runner.bind(pipeline, chunk_ms)

    switch_timer.start()
    runner.run_benchmark()
    switch_timer.stop()

    aggregate = runner.aggregate()
    aggregate["switch_latency_ms"] = summarize(switch_timer.gaps_ms)
    aggregate["stability"]["switches_completed"] = len(switch_timer.gaps_ms)
    hardware = collect_hardware_info()
    report = finalize_run(run, aggregate, targets, hardware=hardware)
    print(report)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
