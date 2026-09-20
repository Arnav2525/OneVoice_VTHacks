

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from verify_retinaface_alignment import RetinaFaceAligner, LIP_PATCH_SIZE

SAMPLE_RATE = 16_000
WINDOW_S = 1.5
WINDOW_SAMPLES = int(WINDOW_S * SAMPLE_RATE)
MOUTH_FPS = 25
WINDOW_FRAMES = int(WINDOW_S * MOUTH_FPS)

def _load_wav_float32(path: Path) -> np.ndarray:
    import wave
    import array

    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    return np.asarray(array.array("h", raw), dtype=np.float32) / 32768.0

def _load_video_frames(path: str, n: int) -> list:
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path}")
    return frames

def dolphin_worker(model, device, mix_slice, mouth_zeros, n_calls, times_out, stop_event):
    from bench._dolphin import infer_separation

    for _ in range(n_calls):
        if stop_event.is_set():
            break
        t0 = time.perf_counter()
        infer_separation(model, mix_slice, mouth_zeros, device=device, fp16=False)
        times_out.append(time.perf_counter() - t0)

def retinaface_worker(aligner, frames, n_calls, times_out, stop_event):
    i = 0
    for _ in range(n_calls):
        if stop_event.is_set():
            break
        frame = frames[i % len(frames)]
        i += 1
        _, dt = aligner.align_frame(frame)
        times_out.append(dt)

def summarize(label: str, times: list[float]) -> None:
    if not times:
        print(f"{label}: no samples")
        return
    arr = np.asarray(times) * 1000.0
    print(f"{label}: n={len(arr)} mean={arr.mean():.1f}ms median={np.median(arr):.1f}ms "
          f"p95={np.percentile(arr,95):.1f}ms max={arr.max():.1f}ms")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", required=True, help="TEST_DIR (corpus tt/ dir, has corpus_mix_01/mix.wav)")
    parser.add_argument("--video", required=True, help="talking-head video for RetinaFace timing")
    parser.add_argument("--n-dolphin", type=int, default=6, help="# Dolphin window inferences per phase")
    parser.add_argument("--n-retinaface", type=int, default=40, help="# RetinaFace align_frame calls per phase")
    args = parser.parse_args()

    from bench._dolphin import load_dolphin

    print("Loading real Dolphin model...")
    model, device, stats = load_dolphin(device=None, warmup=True)
    print(f"  loaded on {device}: {stats.param_count} params")

    print("Loading RetinaFace + FAN aligner...")
    aligner = RetinaFaceAligner()

    mix_path = Path(args.test_dir) / "corpus_mix_01" / "mix.wav"
    mixture = _load_wav_float32(mix_path)
    mix_slice = mixture[:WINDOW_SAMPLES]
    mouth_zeros = np.zeros((WINDOW_FRAMES, LIP_PATCH_SIZE, LIP_PATCH_SIZE), dtype=np.float32)

    frames = _load_video_frames(args.video, args.n_retinaface)
    print(f"loaded {len(frames)} video frames for RetinaFace timing")

    print("\n--- Phase A: solo baselines ---")
    dolphin_solo: list[float] = []
    stop = threading.Event()
    dolphin_worker(model, device, mix_slice, mouth_zeros, args.n_dolphin, dolphin_solo, stop)
    summarize("Dolphin solo", dolphin_solo)

    retinaface_solo: list[float] = []
    retinaface_worker(aligner, frames, args.n_retinaface, retinaface_solo, stop)
    summarize("RetinaFace+FAN solo", retinaface_solo)

    print("\n--- Phase B: concurrent (both running at once, real GPU contention) ---")
    dolphin_concurrent: list[float] = []
    retinaface_concurrent: list[float] = []
    stop2 = threading.Event()
    t_dolphin = threading.Thread(
        target=dolphin_worker,
        args=(model, device, mix_slice, mouth_zeros, args.n_dolphin, dolphin_concurrent, stop2),
    )
    t_retinaface = threading.Thread(
        target=retinaface_worker,
        args=(aligner, frames, args.n_retinaface, retinaface_concurrent, stop2),
    )
    wall_start = time.perf_counter()
    t_dolphin.start()
    t_retinaface.start()
    t_dolphin.join()
    t_retinaface.join()
    wall_elapsed = time.perf_counter() - wall_start
    summarize("Dolphin concurrent", dolphin_concurrent)
    summarize("RetinaFace+FAN concurrent", retinaface_concurrent)
    print(f"wall-clock for concurrent phase: {wall_elapsed:.1f}s")

    print("\n" + "=" * 68)
    print("GPU CONTENTION RESULT")
    print("=" * 68)
    if dolphin_solo and dolphin_concurrent:
        d_solo = np.mean(dolphin_solo) * 1000.0
        d_conc = np.mean(dolphin_concurrent) * 1000.0
        d_pct = (d_conc / d_solo - 1.0) * 100.0
        print(f"Dolphin per-window compute: {d_solo:.1f}ms solo -> {d_conc:.1f}ms concurrent "
              f"({d_pct:+.0f}%)")
    if retinaface_solo and retinaface_concurrent:
        r_solo = np.mean(retinaface_solo) * 1000.0
        r_conc = np.mean(retinaface_concurrent) * 1000.0
        r_pct = (r_conc / r_solo - 1.0) * 100.0
        print(f"RetinaFace+FAN per-frame:   {r_solo:.1f}ms solo -> {r_conc:.1f}ms concurrent "
              f"({r_pct:+.0f}%)")
        frame_budget_ms = 1000.0 / 25.0
        print(f"25fps frame budget: {frame_budget_ms:.1f}ms -- "
              f"concurrent RetinaFace+FAN is "
              f"{'UNDER' if r_conc < frame_budget_ms else 'OVER'} budget")
        print(f"visual staleness budget (120ms): concurrent RetinaFace+FAN is "
              f"{'UNDER' if r_conc < 120.0 else 'OVER'} that too")
    if dolphin_solo and dolphin_concurrent:
        hop_ms = 750.0
        print(f"hop budget (750ms): concurrent Dolphin compute is "
              f"{'UNDER' if d_conc < hop_ms else 'OVER'} -- "
              f"{'compute still hides behind the hop wait' if d_conc < hop_ms else 'compute-hiding trick BREAKS under contention'}")
    print("=" * 68)

if __name__ == "__main__":
    main()
