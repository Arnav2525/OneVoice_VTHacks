

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from verify_retinaface_alignment import RetinaFaceAligner

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="talking-head video (any cv2-readable format)")
    parser.add_argument("--n-frames", type=int, default=50)
    args = parser.parse_args()

    import cv2
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch device: {device}")

    cap = cv2.VideoCapture(args.video)
    frames = []
    for _ in range(args.n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {args.video}")
    print(f"loaded {len(frames)} frames from {args.video}")

    print("building RetinaFaceAligner (loads RetinaFace + FAN weights)...")
    aligner = RetinaFaceAligner()

    print(f"timing {len(frames)} align_frame() calls...")
    timings = []
    for i, frame in enumerate(frames):
        _, dt = aligner.align_frame(frame)
        timings.append(dt)
        if i == 0:
            print(f"  frame 0 (includes model warmup): {dt*1000:.1f}ms")

    steady = timings[1:] if len(timings) > 1 else timings
    arr = np.asarray(steady) * 1000.0
    frame_budget_ms = 1000.0 / 25.0

    print("\n" + "=" * 60)
    print("RETINAFACE+FAN GPU TIMING RESULT")
    print("=" * 60)
    print(f"device: {device}")
    print(f"n frames (steady-state, excl. cold start): {len(arr)}")
    print(f"mean:   {arr.mean():.1f} ms")
    print(f"median: {np.median(arr):.1f} ms")
    print(f"p95:    {np.percentile(arr, 95):.1f} ms")
    print(f"max:    {arr.max():.1f} ms")
    print(f"25fps live frame budget: {frame_budget_ms:.1f} ms")
    feasible = arr.mean() < frame_budget_ms
    print(f"mean vs budget: {'UNDER budget (feasible)' if feasible else 'OVER budget (not feasible without frame-skipping)'}")
    print(f"p95 vs budget:  {'UNDER budget' if np.percentile(arr, 95) < frame_budget_ms else 'OVER budget'}")
    print("=" * 60)
    print(
        "\nNote: this is RetinaFace+FAN alone, on an otherwise-idle GPU. In the "
        "real live pipeline it would run concurrently with Dolphin's own "
        "compute (244ms measured per 1.5s window on T4) and needs to "
        "fit inside the visual staleness budget (visual_max_age_ms, default "
        "120ms) on every single frame -- treat this number as a "
        "best case, not a guaranteed live number."
    )

if __name__ == "__main__":
    main()
