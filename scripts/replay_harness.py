

from __future__ import annotations

import argparse
import array
import logging
import sys
import time
import wave
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from onevoice.audio.io import RecordingAudioSource, WavFileSink
from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.frame import Frame
from onevoice.streaming.app import (
    build_face_tracker,
    build_separator,
    build_target_selector,
    load_experiment_config,
)
from onevoice.streaming.pipeline import StreamingPipeline

logger = logging.getLogger("onevoice.replay_harness")

def _now_ms() -> float:
    return time.monotonic() * 1000.0

class ReplayAudioSource:

    def __init__(self, path: str | Path, chunk_samples: int = 320) -> None:
        with wave.open(str(path), "rb") as w:
            self.sample_rate = w.getframerate()
            self.channels = w.getnchannels()
            n = w.getnframes()
            raw = w.readframes(n)
        pcm16 = array.array("h", raw)
        self._samples = array.array("f", (x / 32768.0 for x in pcm16))
        self.duration_s = len(self._samples) / (self.sample_rate * max(1, self.channels))
        self._chunk_samples = chunk_samples
        self._pos = 0
        self._running = False
        self.exhausted = False

    def start(self) -> None:
        self._running = True
        self._pos = 0
        self.exhausted = False

    def stop(self) -> None:
        self._running = False

    def read(self) -> AudioChunk:
        if not self._running:
            raise RuntimeError("ReplayAudioSource is not running")
        time.sleep(self._chunk_samples / self.sample_rate)
        n_vals = self._chunk_samples * self.channels
        end = self._pos + n_vals
        chunk = self._samples[self._pos : end]
        if len(chunk) < n_vals:
            self.exhausted = True
            chunk = chunk + array.array("f", [0.0] * (n_vals - len(chunk)))
        self._pos = end
        return AudioChunk(
            timestamp_ms=_now_ms(),
            data=chunk,
            sample_rate=self.sample_rate,
            channels=self.channels,
            metadata={"exhausted": self.exhausted, "replay": True},
        )

class ReplayVideoSource:

    def __init__(self, path: str | Path, fps: float = 25.0) -> None:
        import cv2

        cap = cv2.VideoCapture(str(path))
        self._frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self._frames.append(frame)
        cap.release()
        if not self._frames:
            raise RuntimeError(f"no frames read from {path}")
        self.fps = fps
        self._idx = 0
        self._running = False

    def start(self) -> None:
        self._running = True
        self._idx = 0

    def stop(self) -> None:
        self._running = False

    def read(self) -> Frame:
        if not self._running:
            raise RuntimeError("ReplayVideoSource is not running")
        time.sleep(1.0 / self.fps)
        frame_bgr = self._frames[min(self._idx, len(self._frames) - 1)]
        self._idx += 1
        h, w = frame_bgr.shape[:2]

        return Frame(
            timestamp_ms=_now_ms(),
            data=frame_bgr,
            metadata={"width": w, "height": h, "replay": True},
        )

def build_replay_pipeline(
    config: dict[str, Any],
    video_path: str,
    audio_path: str,
    output_dir: Path,
) -> tuple[StreamingPipeline, ReplayAudioSource, float]:
    audio = config.get("audio", {})
    chunk_samples = int(audio.get("chunk_samples", 320))

    base_audio_source = ReplayAudioSource(audio_path, chunk_samples=chunk_samples)
    sample_rate = base_audio_source.sample_rate
    channels = base_audio_source.channels
    chunk_duration_ms = 1000.0 * chunk_samples / sample_rate

    video = config.get("video", {})
    fps = float(video.get("replay_fps", 25.0))
    video_source = ReplayVideoSource(video_path, fps=fps)

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_source = RecordingAudioSource(
        base_audio_source,
        WavFileSink(output_dir / "mixture_before.wav", sample_rate, channels),
    )
    audio_sink = WavFileSink(output_dir / "separated_after.wav", sample_rate, channels)

    pipeline = StreamingPipeline(
        audio_source=audio_source,
        audio_sink=audio_sink,
        video_source=video_source,
        face_tracker=build_face_tracker(config),
        target_selector=build_target_selector(config),
        target_separator=build_separator(config),
    )
    return pipeline, base_audio_source, chunk_duration_ms

def _format_status(status: dict[str, Any]) -> str:
    return (
        f"backend={status.get('backend')} device={status.get('device')} "
        f"real_separation={status.get('is_real_separation')} "
        f"avg_infer={status.get('avg_infer_ms', 0.0):.1f}ms "
        f"failures={status.get('failures')}"
    )

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="video file path (any cv2-readable format)")
    parser.add_argument("--audio", required=True, help="WAV file path (the mixture to feed in)")
    parser.add_argument("--config", default=None, help="experiment YAML (backend/video/selection)")
    parser.add_argument("--output-dir", required=True, help="where to write recorded WAVs")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="run seconds (default: audio clip duration + 5s tail for buffered non-causal drain)",
    )
    parser.add_argument("--telemetry-interval", type=float, default=2.0)
    parser.add_argument(
        "--no-prewarm",
        action="store_true",
        help=(
            "skip pre-warming the backend before the timed clock starts (default: "
            "prewarm). Without prewarm, cold model load (~8-11s for Dolphin) eats "
            "into the clip's real-time budget and can leave no window's-worth of "
            "audio for real separation to ever run on a short clip -- this is a "
            "test-harness artifact, not a pipeline bug. A real deployed app loads "
            "the model once at startup, matching --prewarm (the default), not "
            "--no-prewarm."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = load_experiment_config(args.config)
    output_dir = Path(args.output_dir)
    pipeline, audio_source, chunk_ms = build_replay_pipeline(
        config, args.video, args.audio, output_dir
    )
    duration = args.duration if args.duration is not None else audio_source.duration_s + 5.0
    logger.info(
        "chunk duration: %.1f ms | source clip: %.1fs | run duration: %.1fs (audio+tail)",
        chunk_ms,
        audio_source.duration_s,
        duration,
    )

    prewarm_elapsed_s = 0.0
    if not args.no_prewarm:
        load_fn = getattr(pipeline.target_separator, "load", None)
        if callable(load_fn):
            logger.info("pre-warming backend before starting the timed clock...")
            prewarm_start = time.monotonic()
            try:
                load_fn()
            except Exception:  # noqa: BLE001 - fall through to lazy load on first chunk
                logger.exception("prewarm failed; backend will cold-load on first chunk instead")
            else:
                prewarm_elapsed_s = time.monotonic() - prewarm_start
                logger.info("prewarm complete in %.1fs", prewarm_elapsed_s)

    try:
        pipeline.start()
    except Exception:  # noqa: BLE001 - never let startup crash the process
        logger.exception("failed to start pipeline")
        return 1

    wall_start = time.monotonic()
    try:
        deadline = wall_start + duration
        interval = max(0.1, args.telemetry_interval)
        while pipeline.is_running:
            time.sleep(interval)
            status = pipeline.get_backend_status()
            logger.info("telemetry | %s", _format_status(status))
            if time.monotonic() >= deadline:
                break
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down")
    finally:
        pipeline.stop()
        close = getattr(pipeline.target_separator, "close", None)
        if callable(close):
            close()
    wall_elapsed = time.monotonic() - wall_start

    stats = pipeline.get_stats()
    kept_up = wall_elapsed <= audio_source.duration_s + 5.0 + 2.0
    any_drops = any(
        stats.get(k, 0) for k in ("audio_drops", "video_drops", "playback_drops")
    )

    separated_path = output_dir / "separated_after.wav"
    sep_rms = _wav_rms(separated_path) if separated_path.is_file() else None
    is_silent = sep_rms is not None and sep_rms < 1.0

    lock_state = None
    selector = getattr(pipeline, "_target_selector", None)
    get_state = getattr(selector, "get_state", None)
    if callable(get_state):
        try:
            lock_state = get_state()
        except Exception:  # noqa: BLE001 - diagnostic only, never fail the run over this
            lock_state = None

    print("\n" + "=" * 60)
    print("REPLAY HARNESS RESULT")
    print("=" * 60)
    print(f"source clip duration:  {audio_source.duration_s:.1f}s")
    print(f"prewarm time (excluded from timed run): {prewarm_elapsed_s:.1f}s")
    print(f"wall-clock runtime:    {wall_elapsed:.1f}s")
    print(f"results processed:     {stats.get('results_processed')}")
    print(f"audio/video/playback drops: {stats.get('audio_drops')}/{stats.get('video_drops')}/{stats.get('playback_drops')}")
    print(f"kept up with real-time: {kept_up and not any_drops}")
    print(f"separated output RMS (int16 scale): {sep_rms if sep_rms is not None else 'n/a'}")
    print(f"target-selector lock state at stop: {lock_state if lock_state is not None else 'n/a (selector has no get_state())'}")
    print(f"recorded: {output_dir / 'mixture_before.wav'}")
    print(f"recorded: {output_dir / 'separated_after.wav'}")
    print("=" * 60)
    if not kept_up or any_drops:
        print(
            "\nNOTE: pipeline fell behind real-time and/or dropped queued work -- "
            "this is the honest signal a Kaggle T4 offline-benchmark projection "
            "cannot give you. Re-check on the real target GPU."
        )
    if is_silent:

        if lock_state is not None:
            print(
                "\nWARNING: separated_after.wav is silent (RMS ~0) despite the pipeline "
                "reporting real-time keep-up -- keep-up only measures queue/thread cadence, "
                "not whether a target was ever acquired. The active-speaker lock "
                f"(state: {lock_state}) likely never crossed acquire_thresh, so "
                "no_target_policy: silence suppressed output for the whole clip. This is "
                "NOT the same failure mode as the crop-quality issues found elsewhere -- "
                "it means the motion-based ASD heuristic "
                "(src/onevoice/video/asd/motion_scorer.py) never scored this footage's "
                "mouth motion above acquire_thresh, independent of separation quality."
            )
        else:
            print(
                "\nWARNING: separated_after.wav is silent (RMS ~0) despite the pipeline "
                "reporting real-time keep-up. The active selector has no get_state() "
                "(first_track, or a manually-driven selector like the tap-to-select "
                "demo) -- there is no acquire_thresh/ASD lock to blame here, do NOT "
                "assume that explanation. Check, in order: (1) STDERR above for "
                "'loaded on cuda'/'warmed up' timestamps -- if they land close to or "
                "past the clip duration, the model never finished becoming ready "
                "before the clip ended; if you "
                "passed --no-prewarm this is the likely cause and the fix is to drop "
                "that flag or use a longer clip; (2) whether any face was ever "
                "detected/tracked at all in this footage; (3) for a manually-driven "
                "selector, whether set_manual_target() was ever actually called "
                "(e.g. no click happened) -- no_target_policy: silence is the "
                "correct, honest behavior in that case, not a bug."
            )
    return 0

def _wav_rms(path: Path) -> float | None:

    import array
    import wave

    try:
        with wave.open(str(path), "rb") as w:
            n = w.getnframes()
            raw = w.readframes(n)
    except Exception:  # noqa: BLE001 - diagnostic only
        return None
    samples = array.array("h", raw)
    if not samples:
        return None
    return (sum(s * s for s in samples) / len(samples)) ** 0.5

if __name__ == "__main__":
    raise SystemExit(main())
