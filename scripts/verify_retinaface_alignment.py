

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

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

STABLE_PNT_IDS = [33, 36, 39, 42, 45]
MOUTH_LM_SLICE = slice(48, 68)
STD_SIZE = (256, 256)
FACE_CROP_SIZE = 224
MOUTH_CROP_HALF = 48

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

def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    dh = int(round((h - size) / 2.0))
    dw = int(round((w - size) / 2.0))
    return img[dh : dh + size, dw : dw + size]

def _face2head(box, scale: float = 1.5) -> list[float]:
    x0, y0, x1, y1 = box[:4]
    width, height = x1 - x0, y1 - y0
    wc, hc = (x1 + x0) / 2.0, (y1 + y0) / 2.0
    s = max(width, height) * scale
    return [wc - s / 2.0, hc - s / 2.0, wc + s / 2.0, hc + s / 2.0]

def _cut_patch(img: np.ndarray, landmarks: np.ndarray, half_h: int, half_w: int) -> np.ndarray:

    cx, cy = np.mean(landmarks, axis=0)
    h, w = img.shape[:2]
    cy = max(cy, half_h)
    cy = min(cy, h - half_h)
    cx = max(cx, half_w)
    cx = min(cx, w - half_w)
    y0, y1 = int(round(cy - half_h)), int(round(cy + half_h))
    x0, x1 = int(round(cx - half_w)), int(round(cx + half_w))
    return img[y0:y1, x0:x1]

class RetinaFaceAligner:

    def __init__(self) -> None:
        from bench._dolphin import ensure_dolphin_import_path

        ensure_dolphin_import_path()
        from face_detection_utils import detect_faces
        import face_alignment
        import torch

        self._detect_faces = detect_faces
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
        )
        mean_face_path = REPO_ROOT / "vendor" / "dolphin" / "assets" / "20words_mean_face.npy"
        self._mean_face = np.load(mean_face_path)
        self._last_good: np.ndarray | None = None

    def _detect_head_box(self, frame_bgr: np.ndarray) -> list[float] | None:

        boxes, _ = self._detect_faces(
            frame_bgr, threshold=0.9, allow_upscaling=False, assume_bgr=True
        )
        if boxes is None or len(boxes) == 0:
            boxes, _ = self._detect_faces(
                frame_bgr, threshold=0.7, allow_upscaling=True, assume_bgr=True
            )
        if boxes is None or len(boxes) == 0:
            return None
        return _face2head(boxes[0], scale=1.5)

    def _pad_crop_resize(self, frame_bgr: np.ndarray, head_box: list[float]) -> np.ndarray | None:

        import cv2
        from PIL import Image

        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        face = img.crop(tuple(head_box)).resize((FACE_CROP_SIZE, FACE_CROP_SIZE))
        return np.asarray(face)

    def _warp_and_crop(self, face_rgb: np.ndarray, landmarks: np.ndarray, tform) -> np.ndarray | None:
        import cv2
        from skimage import transform as sktf

        warped = sktf.warp(face_rgb, inverse_map=tform.inverse, output_shape=STD_SIZE)
        warped = (warped * 255.0).astype(np.uint8)
        trans_landmarks = tform(landmarks)

        mouth_lm = trans_landmarks[MOUTH_LM_SLICE]
        try:
            patch = _cut_patch(warped, mouth_lm, MOUTH_CROP_HALF, MOUTH_CROP_HALF)
        except Exception:  # noqa: BLE001
            return None
        if patch.size == 0:
            return None

        gray = (
            cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
            if patch.ndim == 3
            else patch.astype(np.float32)
        )
        if gray.shape != (96, 96):
            gray = cv2.resize(gray, (96, 96))
        crop88 = _center_crop(gray, LIP_PATCH_SIZE)
        return ((crop88 / 255.0) - _LRW_MEAN) / _LRW_STD

    def align_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:

        from skimage import transform as sktf

        t0 = time.perf_counter()
        head_box = self._detect_head_box(frame_bgr)
        if head_box is None:
            return self._fallback(), time.perf_counter() - t0
        face_rgb = self._pad_crop_resize(frame_bgr, head_box)
        if face_rgb is None:
            return self._fallback(), time.perf_counter() - t0
        preds = self._fa.get_landmarks(face_rgb)
        if not preds:
            return self._fallback(), time.perf_counter() - t0
        landmarks = np.asarray(preds[0], dtype=np.float64)

        src = landmarks[STABLE_PNT_IDS]
        dst = self._mean_face[STABLE_PNT_IDS]
        tform = sktf.estimate_transform("similarity", src, dst)
        normed = self._warp_and_crop(face_rgb, landmarks, tform)
        if normed is None:
            return self._fallback(), time.perf_counter() - t0
        self._last_good = normed
        return normed, time.perf_counter() - t0

    def align_window(self, frames_bgr: list[np.ndarray], window_margin: int = 12) -> np.ndarray:

        from collections import deque
        from skimage import transform as sktf

        n = len(frames_bgr)
        raw_landmarks: list[np.ndarray | None] = [None] * n
        face_crops: list[np.ndarray | None] = [None] * n

        detect_every = 8
        prev_box: list[float] | None = None
        prev_lm: np.ndarray | None = None
        for i, frame_bgr in enumerate(frames_bgr):
            if i % detect_every == 0 or prev_box is None:
                head_box = self._detect_head_box(frame_bgr)
                box = head_box if head_box is not None else prev_box
            else:
                box = prev_box
            if box is None:
                continue
            face_rgb = self._pad_crop_resize(frame_bgr, box)
            if face_rgb is None:
                continue
            preds = self._fa.get_landmarks(face_rgb)
            lm = np.asarray(preds[0], dtype=np.float64) if preds else prev_lm
            face_crops[i] = face_rgb
            raw_landmarks[i] = lm
            prev_box = box
            if lm is not None:
                prev_lm = lm

        first = next(
            (j for j in range(n) if face_crops[j] is not None and raw_landmarks[j] is not None),
            None,
        )
        if first is None:

            return np.zeros((n, LIP_PATCH_SIZE, LIP_PATCH_SIZE), dtype=np.float32)
        for j in range(first):
            face_crops[j] = face_crops[first]
            raw_landmarks[j] = raw_landmarks[first]

        last_valid = raw_landmarks[first]
        for i in range(n):
            if raw_landmarks[i] is None:
                raw_landmarks[i] = last_valid
            else:
                last_valid = raw_landmarks[i]

        outputs: list[np.ndarray | None] = [None] * n
        q_idx: "deque[int]" = deque()
        tform = None
        for i in range(n):
            q_idx.append(i)
            if len(q_idx) == window_margin:
                window_idxs = list(q_idx)
                smoothed = np.mean([raw_landmarks[j] for j in window_idxs], axis=0)
                cur_i = q_idx.popleft()
                src = smoothed[STABLE_PNT_IDS]
                dst = self._mean_face[STABLE_PNT_IDS]
                tform = sktf.estimate_transform("similarity", src, dst)
                face_rgb = face_crops[cur_i]
                if face_rgb is None:
                    outputs[cur_i] = None
                else:
                    outputs[cur_i] = self._warp_and_crop(face_rgb, raw_landmarks[cur_i], tform)

        while q_idx:
            cur_i = q_idx.popleft()
            face_rgb = face_crops[cur_i]
            if face_rgb is None or tform is None:
                outputs[cur_i] = None
            else:
                outputs[cur_i] = self._warp_and_crop(face_rgb, raw_landmarks[cur_i], tform)

        last_good = np.zeros((LIP_PATCH_SIZE, LIP_PATCH_SIZE), dtype=np.float32)
        for i in range(n):
            if outputs[i] is None:
                outputs[i] = last_good
            else:
                last_good = outputs[i]
        return np.stack(outputs, axis=0)  # type: ignore[arg-type]

    def _fallback(self) -> np.ndarray:
        if self._last_good is not None:
            return self._last_good
        return np.zeros((LIP_PATCH_SIZE, LIP_PATCH_SIZE), dtype=np.float32)

def _ceiling_mouth_tensor(npz_path: Path, start_frame: int) -> np.ndarray:
    data = np.load(npz_path)["data"]
    window = data[start_frame : start_frame + WINDOW_FRAMES].astype(np.float32)
    window = window / 255.0
    t, h, w = window.shape
    th = tw = LIP_PATCH_SIZE
    dh = int(round((h - th) / 2.0))
    dw = int(round((w - tw) / 2.0))
    window = window[:, dh : dh + th, dw : dw + tw]
    window = (window - _LRW_MEAN) / _LRW_STD
    return window

def _read_frames(path: Path, n: int) -> list:
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out

def process_clip(clip_id: str, target: str, model, device, aligner, timing) -> dict | None:

    from bench._dolphin import infer_separation
    from onevoice.telemetry.quality import evaluate_pair

    corpus_dir = TT_ROOT / clip_id
    video_path = MOUTH_CACHE_ROOT / clip_id / target / "temp_25fps.mp4"
    npz_path = corpus_dir / f"{target}_mouth.npz"
    mix_path = corpus_dir / "mix.wav"
    ref_path = corpus_dir / f"{target}.wav"
    if not (video_path.is_file() and npz_path.is_file() and mix_path.is_file() and ref_path.is_file()):
        return None

    mixture = _load_wav_float32(mix_path)
    reference = _load_wav_float32(ref_path)
    if WINDOW_SAMPLES > min(len(mixture), len(reference)):
        return None
    mix_slice = mixture[:WINDOW_SAMPLES]
    ref_slice = reference[:WINDOW_SAMPLES]

    frames_bgr = _read_frames(video_path, WINDOW_FRAMES)
    if len(frames_bgr) < WINDOW_FRAMES:
        return None

    mouth_ceiling = _ceiling_mouth_tensor(npz_path, 0)
    causal_frames = []
    for frame_bgr in frames_bgr:
        patch, dt = aligner.align_frame(frame_bgr)
        causal_frames.append(patch)
        timing.append(dt)
    mouth_causal = np.stack(causal_frames, axis=0)
    mouth_smoothed = aligner.align_window(frames_bgr, window_margin=12)

    out: dict = {}
    for label, mouth in (
        ("retinaface_causal", mouth_causal),
        ("retinaface_smoothed", mouth_smoothed),
        ("ceiling", mouth_ceiling),
    ):
        estimate = infer_separation(model, mix_slice, mouth, device=device, fp16=False)
        n = min(len(estimate), len(ref_slice), len(mix_slice))
        metrics = evaluate_pair(ref_slice[:n], estimate[:n], mix_slice[:n], SAMPLE_RATE)
        rms = float(np.sqrt(np.mean(np.square(estimate[:n]))))
        snri = metrics["si_snr_improvement_db"]
        out[label] = (rms, float(snri) if snri is not None else None)
    return out

def main() -> None:
    import argparse

    from bench._dolphin import load_dolphin
    import torch

    global TT_ROOT, MOUTH_CACHE_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda/cpu (default: auto-detect)")
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
    device_arg = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.tt_root:
        TT_ROOT = Path(args.tt_root)
    if args.mouth_cache_root:
        MOUTH_CACHE_ROOT = Path(args.mouth_cache_root)
    print(f"tt_root={TT_ROOT}")
    print(f"mouth_cache_root={MOUTH_CACHE_ROOT}")

    print(f"Loading real Dolphin model ({device_arg}, fp32)...")
    model, device, stats = load_dolphin(device=device_arg, warmup=False)
    print(f"  loaded: {stats.param_count} params")

    print("Loading RetinaFace + FAN aligner (this downloads FAN weights on first use)...")
    aligner = RetinaFaceAligner()

    per_clip: dict[str, dict] = {}
    skipped: list[str] = []
    timing: list[float] = []
    total = len(CLIP_IDS) * len(TARGETS)
    done = 0
    for clip_id in CLIP_IDS:
        for target in TARGETS:
            done += 1
            label = f"{clip_id}/{target}"
            print(f"[{done}/{total}] {label} ...", end=" ", flush=True)
            result = process_clip(clip_id, target, model, device, aligner, timing)
            if result is None:
                print("SKIPPED")
                skipped.append(label)
                continue
            per_clip[label] = result
            print(
                f"causal={result['retinaface_causal'][1]:+.2f}dB "
                f"smoothed={result['retinaface_smoothed'][1]:+.2f}dB "
                f"ceiling={result['ceiling'][1]:+.2f}dB"
            )

    print("\n" + "=" * 72)
    print(f"AGGREGATE ({len(per_clip)}/{total} clips scored, {len(skipped)} skipped)")
    print("=" * 72)
    print("NOTE: 'new' (MediaPipe) not recomputed here -- already validated:")
    print("      mean +11.53 dB / median +15.93 dB / gate 26/32 (verify_lip_roi_real_crop.py)")
    print("NOTE: cross-speaker rescue was tried and reverted. Only causal/smoothed/ceiling remain.")
    print(f"{'condition':<20} {'mean RMS':>10} {'mean SI-SNRi':>14} {'median SI-SNRi':>16} {'gate (>=7dB)':>14}")
    for label in ("retinaface_causal", "retinaface_smoothed", "ceiling"):
        rms_vals = [v[label][0] for v in per_clip.values()]
        snri_vals = [v[label][1] for v in per_clip.values() if v[label][1] is not None]
        mean_rms = float(np.mean(rms_vals)) if rms_vals else float("nan")
        mean_snri = float(np.mean(snri_vals)) if snri_vals else float("nan")
        median_snri = float(np.median(snri_vals)) if snri_vals else float("nan")
        gate_pass = sum(1 for v in snri_vals if v >= GATE_THRESHOLD_DB)
        print(
            f"{label:<20} {mean_rms:>10.6f} {mean_snri:>+14.2f} {median_snri:>+16.2f} "
            f"{gate_pass:>6}/{len(snri_vals)}"
        )
    print("=" * 72)

    if timing:
        arr = np.asarray(timing) * 1000.0
        print(f"\nRetinaFace+FAN per-frame wall-clock (n={len(arr)}): "
              f"mean={arr.mean():.1f}ms p50={np.median(arr):.1f}ms p95={np.percentile(arr,95):.1f}ms "
              f"max={arr.max():.1f}ms")
        print(f"At 25fps live, a frame budget is {1000.0/25:.1f}ms -- "
              f"{'FEASIBLE' if arr.mean() < 1000.0/25 else 'NOT FEASIBLE without frame-skipping/GPU'}.")

    if skipped:
        print(f"\nSkipped clips: {skipped}")

if __name__ == "__main__":
    main()
