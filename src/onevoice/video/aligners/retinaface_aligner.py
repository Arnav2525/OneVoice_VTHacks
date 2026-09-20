

from __future__ import annotations

import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from onevoice.separation.dolphin_loader import ensure_dolphin_import_path, vendor_root

logger = logging.getLogger(__name__)

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

LIP_PATCH_SIZE = 88
_LRW_MEAN = 0.421
_LRW_STD = 0.165

STABLE_PNT_IDS = [33, 36, 39, 42, 45]
MOUTH_LM_SLICE = slice(48, 68)
STD_SIZE = (256, 256)
FACE_CROP_SIZE = 224
MOUTH_CROP_HALF = 48
DETECT_EVERY = 8
DEFAULT_WINDOW_MARGIN = 12

def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    dh = int(round((h - size) / 2.0))
    dw = int(round((w - size) / 2.0))
    return img[dh : dh + size, dw : dw + size]

def _face2head(box: Any, scale: float = 1.5) -> list[float]:
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

    def __init__(self, device: str | None = None) -> None:
        ensure_dolphin_import_path()
        from face_detection_utils import detect_faces
        import face_alignment
        import torch

        self._detect_faces = detect_faces
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, flip_input=False, device=device
        )
        mean_face_path = vendor_root() / "assets" / "20words_mean_face.npy"
        if not mean_face_path.is_file():
            raise FileNotFoundError(
                f"Dolphin mean-face asset missing: {mean_face_path}. "
                "Run: python scripts/vendor_dolphin.py"
            )
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

    def _warp_and_crop(self, face_rgb: np.ndarray, landmarks: np.ndarray, tform: Any) -> np.ndarray | None:
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

    def align_window(
        self, frames_bgr: list[np.ndarray], window_margin: int = DEFAULT_WINDOW_MARGIN
    ) -> np.ndarray:

        from skimage import transform as sktf

        n = len(frames_bgr)
        if n == 0:
            return np.zeros((0, LIP_PATCH_SIZE, LIP_PATCH_SIZE), dtype=np.float32)

        raw_landmarks: list[np.ndarray | None] = [None] * n
        face_crops: list[np.ndarray | None] = [None] * n
        prev_box: list[float] | None = None
        prev_lm: np.ndarray | None = None
        for i, frame_bgr in enumerate(frames_bgr):
            if i % DETECT_EVERY == 0 or prev_box is None:
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
        q_idx: deque[int] = deque()
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
                    outputs[cur_i] = self._warp_and_crop(
                        face_rgb, raw_landmarks[cur_i], tform  # type: ignore[arg-type]
                    )

        while q_idx:
            cur_i = q_idx.popleft()
            face_rgb = face_crops[cur_i]
            if face_rgb is None or tform is None:
                outputs[cur_i] = None
            else:
                outputs[cur_i] = self._warp_and_crop(
                    face_rgb, raw_landmarks[cur_i], tform  # type: ignore[arg-type]
                )

        last_good = np.zeros((LIP_PATCH_SIZE, LIP_PATCH_SIZE), dtype=np.float32)
        for i in range(n):
            if outputs[i] is None:
                outputs[i] = last_good
            else:
                last_good = outputs[i]
                self._last_good = last_good
        return np.stack(outputs, axis=0)  # type: ignore[arg-type]

    def _fallback(self) -> np.ndarray:
        if self._last_good is not None:
            return self._last_good
        return np.zeros((LIP_PATCH_SIZE, LIP_PATCH_SIZE), dtype=np.float32)
