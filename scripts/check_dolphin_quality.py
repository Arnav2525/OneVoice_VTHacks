

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from concurrent.futures import Future
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
import yaml
from scipy.signal import correlate, correlation_lags

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection
from onevoice.separation.adapters.base import AdapterContext
from onevoice.separation.adapters.dolphin_adapter import DolphinAdapter
from onevoice.separation.config import SeparationConfig
from onevoice.separation.dolphin_loader import load_dolphin_model
from onevoice.video.detectors.mediapipe_detector import MediaPipeFaceDetector
from onevoice.video.trackers.iou_tracker import _crop_lip_patch

RATE = 16000

class ImmediateExecutor:
    def submit(self, function, *args):
        result = Future()
        try:
            result.set_result(function(*args))
        except BaseException as exc:
            result.set_exception(exc)
        return result

    def shutdown(self, **kwargs):
        pass

class LatestHopProbe(DolphinAdapter):

    def _integrate_window_result(self, state, window_out, had_visual, window_audio):
        start = state.pending_window_start + self._window_samples - self._hop_samples
        if not state.fifo.size:
            state.fifo_start_sample = start
        state.fifo = np.concatenate((state.fifo, window_out[-self._hop_samples :]))
        state.last_conditioning = "visual" if had_visual else "audio_only"

class RecentHopProbe(DolphinAdapter):

    def _integrate_window_result(self, state, window_out, had_visual, window_audio):
        lookahead = round(self.probe_lookahead * RATE)
        overlap = round(0.02 * RATE)
        end = self._window_samples - lookahead
        start = end - self._hop_samples
        prefix = window_out[start - overlap : start].copy()
        previous = getattr(state, "recent_tail", None)
        if previous is not None:
            alpha = np.linspace(0, 1, overlap, dtype=np.float32)
            prefix = previous * (1 - alpha) + prefix * alpha
        finalized = np.concatenate((prefix, window_out[start : end - overlap]))
        state.recent_tail = window_out[end - overlap : end].copy()
        if not state.fifo.size:
            state.fifo_start_sample = state.pending_window_start + start - overlap
        state.fifo = np.concatenate((state.fifo, finalized))
        state.last_conditioning = "visual" if had_visual else "audio_only"

def prepare_speaker(folder, output):
    cache = output / f"{folder.name}_prepared.npz"
    if cache.exists():
        saved = np.load(cache)
        return saved["audio"], saved["patches"]
    detector = MediaPipeFaceDetector()
    audio_parts, patches = [], []
    video_writer = None
    try:
        for path in sorted(folder.glob("*.mpg")):
            capture = cv2.VideoCapture(str(path))
            frame_count = 0
            if abs(capture.get(cv2.CAP_PROP_FPS) - 25) > 0.1:
                raise RuntimeError(f"Expected 25 fps GRID video: {path}")
            try:
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    if video_writer is None:
                        video_writer = cv2.VideoWriter(
                            str(output / f"{folder.name}_video.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            25,
                            (image.shape[1], image.shape[0]),
                        )
                    video_writer.write(image)
                    frame = Frame(len(patches) * 40.0, image, {})
                    found = detector.detect(frame)
                    if len(found) != 1 or found[0].lip_bbox is None:
                        raise RuntimeError(
                            f"Need one visible face: {path}, frame {frame_count}"
                        )
                    patch = _crop_lip_patch(frame, found[0].lip_bbox)
                    if patch is None:
                        raise RuntimeError(f"Missing mouth patch in {path}")
                    patches.append(patch)
                    frame_count += 1
            finally:
                capture.release()
            raw = subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-vn",
                    "-af",
                    "aresample=async=1:first_pts=0",
                    "-ar",
                    str(RATE),
                    "-ac",
                    "1",
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                check=True,
                capture_output=True,
            ).stdout
            audio = np.frombuffer(raw, dtype="<f4")
            length = frame_count * RATE // 25
            audio_parts.append(np.pad(audio, (0, max(0, length - len(audio))))[:length])
            print(f"Prepared {path.name}: {frame_count} frames", flush=True)
    finally:
        detector.close()
        if video_writer is not None:
            video_writer.release()
    audio, patches = np.concatenate(audio_parts), np.asarray(patches)
    np.savez_compressed(cache, audio=audio, patches=patches)
    return audio, patches

def stream(
    adapter_class,
    config,
    model,
    mixture,
    patches,
    fps,
    *,
    paced=False,
    audio_config=None,
):
    adapter = adapter_class(config)
    if not paced:
        adapter._executor.shutdown(wait=True)
        adapter._executor = ImmediateExecutor()
    processors = []
    audio_config = audio_config or {}
    if audio_config.get("cleanup", {}).get("enabled"):
        from onevoice.audio.cleanup import RumbleFilter

        processors.append(RumbleFilter(audio_config["cleanup"].get("cutoff_hz", 80)))
    if audio_config.get("denoise", {}).get("enabled"):
        from onevoice.audio.denoise import GtcrnDenoiser

        processors.append(GtcrnDenoiser())
    ctx = AdapterContext("cuda", config)
    runtime, outputs, metadata = [], [], []
    original = adapter._infer_window_forward

    def measured(*args):
        started = time.perf_counter()
        result = original(*args)
        runtime.append(1000 * (time.perf_counter() - started))
        return result

    adapter._infer_window_forward = measured
    padded = np.pad(mixture, (0, 3 * RATE))
    clock_start = time.perf_counter()
    try:
        for frame_index in range(int(len(padded) * fps / RATE)):
            start, end = (
                round(frame_index * RATE / fps),
                round((frame_index + 1) * RATE / fps),
            )
            patch = patches[min(int(frame_index * 25 / fps), len(patches) - 1)]
            timestamp = 1000 * start / RATE
            if paced:
                time.sleep(max(0.0, clock_start + end / RATE - time.perf_counter()))
            track = SpeakerTrack(
                "target",
                (0, 0, 100, 100),
                1.0,
                {
                    "lip_patch": patch,
                    "lip_roi_ts_ms": 1000 * frame_index / fps,
                    "ui_selection_epoch": 1,
                },
            )
            chunk = AudioChunk(timestamp, padded[start:end], RATE, 1, {})
            target = TargetSelection(timestamp, track)
            value = adapter.to_backend(model, chunk, target, [track], ctx)
            output = adapter.from_backend(adapter.infer(model, value, ctx), chunk, ctx)
            eligible = (
                output.metadata.get("output_ready")
                and output.metadata.get("conditioning") == "visual"
            )
            for processor in processors:
                if eligible:
                    output = processor.process(output)
                else:
                    processor.reset()
            outputs.append(
                np.asarray(output.data, dtype=np.float32)
                if eligible
                else np.zeros(end - start, dtype=np.float32)
            )
            metadata.append(output.metadata)
            if not paced:
                adapter.flush_pending("target")
    finally:
        adapter.flush_pending("target")
        adapter.shutdown()
    return np.concatenate(outputs), {
        "forward_p50_ms": float(np.median(runtime)),
        "forward_p95_ms": float(np.percentile(runtime, 95)),
        "forward_calls": len(runtime),
        "ready_chunk_fraction": float(np.mean([m["output_ready"] for m in metadata])),
        "last_metadata": metadata[-1],
    }

def si_sdr(estimate, target):
    estimate, target = estimate - estimate.mean(), target - target.mean()
    projected = target * (estimate @ target) / max(target @ target, 1e-20)
    return float(
        10
        * np.log10(
            (projected @ projected + 1e-20)
            / ((estimate - projected) @ (estimate - projected) + 1e-20)
        )
    )

def score(output, clean, other, mixture):

    correlation = correlate(output, clean, mode="full", method="fft")
    lags = correlation_lags(len(output), len(clean))
    allowed = (lags >= RATE // 4) & (lags <= 3 * RATE)
    lag = int(lags[allowed][np.argmax(np.abs(correlation[allowed]))])
    aligned = output[lag : lag + len(clean)]

    region = slice(3 * RATE // 2, len(clean) - RATE)
    estimate = aligned[region].astype(np.float64)
    target, interference, mix = [
        x[region].astype(np.float64) for x in (clean, other, mixture)
    ]
    if np.sqrt(np.mean(estimate**2)) < 1e-8:
        raise RuntimeError("Output is silent; quality cannot be scored")
    sources = np.stack(
        [target - target.mean(), interference - interference.mean()], axis=1
    )
    gains = np.linalg.lstsq(sources, estimate - estimate.mean(), rcond=None)[0]
    selected_power = np.sum((sources[:, 0] * gains[0]) ** 2)
    unwanted_power = np.sum((sources[:, 1] * gains[1]) ** 2)
    sir = 10 * np.log10((selected_power + 1e-20) / (unwanted_power + 1e-20))
    input_sir = 10 * np.log10(np.sum(target**2) / np.sum(interference**2))
    return aligned, {
        "delay_samples": lag,
        "alignment_delay_ms": 1000 * lag / RATE,
        "evaluation_seconds": [1.5, len(clean) / RATE - 1],
        "input_si_sdr_db": si_sdr(mix, target),
        "output_si_sdr_db": si_sdr(estimate, target),
        "si_sdr_improvement_db": si_sdr(estimate, target) - si_sdr(mix, target),
        "projected_interferer_rejection_improvement_db": float(sir - input_sir),
        "target_projection_gain": float(gains[0]),
        "interferer_projection_gain": float(gains[1]),
        "output_rms": float(np.sqrt(np.mean(estimate**2))),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "runs/quality_check/grid")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/quality_check/grid_result"
    )
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "runs/quality_check/baseline_adapter.py"
    )
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--probe-latest", action="store_true")
    parser.add_argument("--probe-recent", action="store_true")
    parser.add_argument(
        "--config", type=Path, help="Evaluate the shipped config instead of probes"
    )
    parser.add_argument("--third-speaker", choices=["s2"])
    parser.add_argument(
        "--paced",
        action="store_true",
        help="Use real async inference with clock-paced file input",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    speakers = ["s1", "s5"] + ([args.third_speaker] if args.third_speaker else [])
    prepared = [prepare_speaker(args.data / name, args.output) for name in speakers]
    length = min(len(value[0]) for value in prepared)
    clean = [
        audio[:length] * (0.08 / max(np.sqrt(np.mean(audio[:length] ** 2)), 1e-10))
        for audio, _ in prepared
    ]
    scale = min(1.0, 0.95 / max(np.max(np.abs(sum(clean))), 1e-10))
    clean = [value * scale for value in clean]
    mixture = sum(clean)
    sf.write(args.output / "mixture.wav", mixture, RATE, subtype="FLOAT")
    for name, audio in zip(speakers, clean):
        sf.write(args.output / f"{name}_reference.wav", audio, RATE, subtype="FLOAT")
    spec = importlib.util.spec_from_file_location("quality_baseline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    config = SeparationConfig(
        name="dolphin",
        device="cuda",
        precision="float32",
        params={
            "window_s": 2.0,
            "hop_s": 0.75,
            "auto_gain": False,
            "visual_aligner": "mediapipe",
        },
    )
    audio_config = {}
    if args.config:
        settings = yaml.safe_load(args.config.read_text())
        config = SeparationConfig.from_raw(settings["backend"])
        audio_config = settings.get("audio", {})
    print("Loading cached Dolphin on CUDA...", flush=True)
    model = load_dolphin_model(config, "cuda")
    warm = DolphinAdapter(config)
    warm._infer_window_forward(
        model,
        mixture[: 2 * RATE],
        list(prepared[0][1][:50]),
        AdapterContext("cuda", config),
    )
    warm.shutdown()
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "fps": args.fps,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "new_sha256": hashlib.sha256(
            (ROOT / "src/onevoice/separation/adapters/dolphin_adapter.py").read_bytes()
        ).hexdigest(),
        "method": ("Clean GRID voices, equal-RMS mixture, "
                   "MediaPipe live crops, playback gating"),
        "speakers": speakers,
        "paced_async_inference": args.paced,
        "backend_params": config.params,
        "postfilters": audio_config,
        "limitations": (
            "Small read-speech corpus, no room noise/reverb or hardware. "
            "Projection rejection ignores distortion; SI-SDR includes it. "
            "Measured file delay excludes hardware and acoustic latency."
        ),
        "results": {},
    }
    variants = [("baseline", baseline.DolphinAdapter), ("corrected", DolphinAdapter)]
    if args.probe_latest:
        variants.append(("latest_hop_probe", LatestHopProbe))
    if args.probe_recent:
        config = config.model_copy(update={"params": {**config.params, "hop_s": 0.5}})
        variants = [
            (
                "recent_250ms",
                type("Recent250", (RecentHopProbe,), {"probe_lookahead": 0.25}),
            ),
            (
                "recent_500ms",
                type("Recent500", (RecentHopProbe,), {"probe_lookahead": 0.5}),
            ),
        ]
    if args.config:
        variants = [("configured", DolphinAdapter)]
    for version, adapter_class in variants:
        for index, name in enumerate(("s1", "s5")):
            output, performance = stream(
                adapter_class,
                config,
                model,
                mixture,
                prepared[index][1],
                args.fps,
                paced=args.paced,
                audio_config=audio_config,
            )
            aligned, metrics = score(
                output, clean[index], mixture - clean[index], mixture
            )
            metrics.update(performance)
            result["results"][f"{version}_{name}"] = metrics
            sf.write(
                args.output / f"{version}_{name}_raw.wav", output, RATE, subtype="FLOAT"
            )
            sf.write(
                args.output / f"{version}_{name}_aligned.wav",
                aligned,
                RATE,
                subtype="FLOAT",
            )
            metrics_file = (
                "recent_metrics.json" if args.probe_recent else "metrics.json"
            )
            (args.output / metrics_file).write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print(
                f"{version}/{name}: SI-SDRi={metrics['si_sdr_improvement_db']:.2f} dB; "
                f"projection rejection="
                f"{metrics['projected_interferer_rejection_improvement_db']:.2f} dB; "
                f"forward p95={metrics['forward_p95_ms']:.0f} ms",
                flush=True,
            )

if __name__ == "__main__":
    main()
