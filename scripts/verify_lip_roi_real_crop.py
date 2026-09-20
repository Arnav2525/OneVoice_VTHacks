

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

TT_ROOT = REPO_ROOT / "data" / "dolphin_tier_a" / "tt"
MOUTH_CACHE_ROOT = REPO_ROOT / "data" / "dolphin_tier_a" / "mouth_cache"

SAMPLE_RATE = 16_000
MOUTH_FPS = 25
LIP_PATCH_SIZE = 88
WINDOW_S = 1.5
WINDOW_SAMPLES = int(WINDOW_S * SAMPLE_RATE)
WINDOW_FRAMES = int(WINDOW_S * MOUTH_FPS)
GATE_THRESHOLD_DB = 7.0

_LRW_MEAN = 0.421
_LRW_STD = 0.165

CLIP_IDS = [f"corpus_mix_{i:02d}" for i in range(1, 17)]
TARGETS = ("s1", "s2")

def _load_wav_float32(path: Path) -> np.ndarray:
    import wave
    import array

    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    return np.asarray(array.array("h", raw), dtype=np.float32) / 32768.0

def _new_pipeline_mouth_tensor(frames_bgr: list[np.ndarray], tracker) -> np.ndarray:

    from onevoice.core.models.frame import Frame

    out = []
    last_patch = None
    for i, frame_bgr in enumerate(frames_bgr):
        h, w = frame_bgr.shape[:2]

        frame = Frame(
            timestamp_ms=float(i * 1000.0 / MOUTH_FPS),
            data=frame_bgr,
            metadata={"width": w, "height": h},
        )
        tracks = tracker.process_frame(frame)
        if tracks:
            patch = tracks[0].metadata.get("lip_patch")
            if patch is not None:
                last_patch = np.asarray(patch, dtype=np.float32)
        out.append(
            last_patch
            if last_patch is not None
            else np.zeros((LIP_PATCH_SIZE, LIP_PATCH_SIZE), dtype=np.float32)
        )
    return np.stack(out, axis=0)

def _ceiling_mouth_tensor(npz_path: Path, start_frame: int) -> np.ndarray:

    data = np.load(npz_path)["data"]
    window = data[start_frame : start_frame + WINDOW_FRAMES].astype(np.float32)

    window = window / 255.0
    t, h, w = window.shape
    th, tw = LIP_PATCH_SIZE, LIP_PATCH_SIZE
    dh = int(round((h - th) / 2.0))
    dw = int(round((w - tw) / 2.0))
    window = window[:, dh : dh + th, dw : dw + tw]
    window = (window - _LRW_MEAN) / _LRW_STD
    return window

def _find_face_window(frames_bgr: list[np.ndarray], tracker_factory) -> tuple[int, tuple] | None:

    from onevoice.core.models.frame import Frame

    n = len(frames_bgr)
    for start in range(0, max(1, n - WINDOW_FRAMES + 1), 5):
        tracker = tracker_factory()
        lip_bbox = None
        for i in range(start, min(start + WINDOW_FRAMES, n)):
            frame_bgr = frames_bgr[i]
            h, w = frame_bgr.shape[:2]

            frame = Frame(
                timestamp_ms=float(i * 1000.0 / MOUTH_FPS),
                data=frame_bgr,
                metadata={"width": w, "height": h},
            )
            tracks = tracker.process_frame(frame)
            if tracks and "lip_roi" in tracks[0].metadata:
                lip_bbox = tracks[0].metadata["lip_roi"]
                break
        if lip_bbox is not None and start + WINDOW_FRAMES <= n:
            return start, lip_bbox
    return None

def process_clip(clip_id: str, target: str, model, device, detector_factory) -> dict | None:
    from bench._dolphin import infer_separation
    from onevoice.telemetry.quality import evaluate_pair
    from onevoice.video.trackers.iou_tracker import IouFaceTracker, TrackerConfig
    import cv2

    corpus_dir = TT_ROOT / clip_id
    video_path = MOUTH_CACHE_ROOT / clip_id / target / "temp_25fps.mp4"
    npz_path = corpus_dir / f"{target}_mouth.npz"
    mix_path = corpus_dir / "mix.wav"
    ref_path = corpus_dir / f"{target}.wav"
    if not (video_path.is_file() and npz_path.is_file() and mix_path.is_file() and ref_path.is_file()):
        return None

    cap = cv2.VideoCapture(str(video_path))
    frames_bgr = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()
    if len(frames_bgr) < WINDOW_FRAMES:
        return None

    def tracker_factory():
        return IouFaceTracker(detector_factory(), TrackerConfig(min_hits=1))

    found = _find_face_window(frames_bgr, tracker_factory)
    if found is None:
        return None
    start_frame, lip_bbox = found
    start_sample = int(start_frame * SAMPLE_RATE / MOUTH_FPS)
    del lip_bbox

    mixture = _load_wav_float32(mix_path)
    reference = _load_wav_float32(ref_path)
    if start_sample + WINDOW_SAMPLES > min(len(mixture), len(reference)):
        return None
    mix_slice = mixture[start_sample : start_sample + WINDOW_SAMPLES]
    ref_slice = reference[start_sample : start_sample + WINDOW_SAMPLES]
    window_frames_bgr = frames_bgr[start_frame : start_frame + WINDOW_FRAMES]

    mouth_new = _new_pipeline_mouth_tensor(window_frames_bgr, tracker_factory())
    mouth_ceiling = _ceiling_mouth_tensor(npz_path, start_frame)

    out = {}
    for label, mouth in (("new", mouth_new), ("ceiling", mouth_ceiling)):
        estimate = infer_separation(model, mix_slice, mouth, device=device, fp16=False)
        n = min(len(estimate), len(ref_slice), len(mix_slice))
        metrics = evaluate_pair(ref_slice[:n], estimate[:n], mix_slice[:n], SAMPLE_RATE)
        rms = float(np.sqrt(np.mean(np.square(estimate))))
        snri = metrics["si_snr_improvement_db"]
        out[label] = (rms, float(snri) if snri is not None else None)
    return out

def main() -> None:
    import argparse

    from bench._dolphin import load_dolphin
    from onevoice.video.detectors.mediapipe_detector import MediaPipeFaceDetector

    global TT_ROOT, MOUTH_CACHE_ROOT, WINDOW_S, WINDOW_SAMPLES, WINDOW_FRAMES

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto-detect)")
    parser.add_argument("--window-s", type=float, default=WINDOW_S, help="window length in seconds")
    parser.add_argument(
        "--tt-root", default=None,
        help="corpus tt/ dir (mix.wav, s1.wav, s2.wav, *_mouth.npz) -- "
             "default: local data/dolphin_tier_a/tt; on Kaggle pass TEST_DIR",
    )
    parser.add_argument(
        "--mouth-cache-root", default=None,
        help="dir containing <clip>/<spk>/temp_25fps.mp4 raw face videos -- "
             "default: local data/dolphin_tier_a/mouth_cache; on Kaggle pass "
             "the dataset's mouth_cache/ (sibling of DATASET_TT_DIR, not the "
             "rewritten TEST_DIR copy -- these videos need no path rewriting)",
    )
    args = parser.parse_args()
    if args.tt_root:
        TT_ROOT = Path(args.tt_root)
    if args.mouth_cache_root:
        MOUTH_CACHE_ROOT = Path(args.mouth_cache_root)
    WINDOW_S = args.window_s
    WINDOW_SAMPLES = int(WINDOW_S * SAMPLE_RATE)
    WINDOW_FRAMES = int(WINDOW_S * MOUTH_FPS)
    print(f"tt_root={TT_ROOT}")
    print(f"mouth_cache_root={MOUTH_CACHE_ROOT}")
    print(f"window_s={WINDOW_S} ({WINDOW_SAMPLES} samples, {WINDOW_FRAMES} mouth frames)")

    import torch

    device_arg = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading real Dolphin model ({device_arg}, fp32)...")
    model, device, stats = load_dolphin(device=device_arg, warmup=False)
    print(f"  loaded: {stats.param_count} params")

    _detector = MediaPipeFaceDetector()

    def detector_factory():
        return _detector

    per_clip: dict[str, dict] = {}
    skipped: list[str] = []
    total = len(CLIP_IDS) * len(TARGETS)
    done = 0
    for clip_id in CLIP_IDS:
        for target in TARGETS:
            done += 1
            label = f"{clip_id}/{target}"
            print(f"[{done}/{total}] {label} ...", end=" ", flush=True)
            result = process_clip(clip_id, target, model, device, detector_factory)
            if result is None:
                print("SKIPPED (no face window / missing assets)")
                skipped.append(label)
                continue
            per_clip[label] = result
            print(f"new={result['new'][1]:+.2f}dB ceiling={result['ceiling'][1]:+.2f}dB")

    print("\n" + "=" * 72)
    print(f"AGGREGATE ({len(per_clip)}/{total} clips scored, {len(skipped)} skipped) at window_s={WINDOW_S}")
    print("=" * 72)
    print(f"{'condition':<12} {'mean RMS':>10} {'mean SI-SNRi':>14} {'median SI-SNRi':>16} {'gate (>=7dB)':>14}")
    for label in ("new", "ceiling"):
        rms_vals = [v[label][0] for v in per_clip.values()]
        snri_vals = [v[label][1] for v in per_clip.values() if v[label][1] is not None]
        mean_rms = float(np.mean(rms_vals)) if rms_vals else float("nan")
        mean_snri = float(np.mean(snri_vals)) if snri_vals else float("nan")
        median_snri = float(np.median(snri_vals)) if snri_vals else float("nan")
        gate_pass = sum(1 for v in snri_vals if v >= GATE_THRESHOLD_DB)
        print(
            f"{label:<12} {mean_rms:>10.6f} {mean_snri:>+14.2f} {median_snri:>+16.2f} "
            f"{gate_pass:>6}/{len(snri_vals)}"
        )
    print("=" * 72)

    if skipped:
        print(f"\nSkipped clips: {skipped}")

    print(
        f"\nInterpretation: 'new' (live MediaPipe crop, real production path) vs\n"
        f"'ceiling' (Dolphin's own training-distribution input) at window_s={WINDOW_S} --\n"
        "the closer 'new' tracks 'ceiling' here, the more of the offline window-length\n"
        "gain (window_s=2.0's 31/32 offline number, vs 1.5's 28/32) survives on the live crop."
    )

if __name__ == "__main__":
    main()
