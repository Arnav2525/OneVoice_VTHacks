

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

from onevoice.core.models.frame import Frame
from onevoice.video.detectors.base import Detection

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_MODEL_NAME = "blaze_face_short_range.tflite"

try:  # pragma: no cover - exercised only where mediapipe is installed
    import mediapipe as mp
    from mediapipe.tasks.python.core import base_options as mp_base_options
    from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
    from mediapipe.tasks.python.vision.core import image as mp_image
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_running_mode
except ImportError:  # pragma: no cover - optional dependency
    mp = None  # type: ignore[assignment]
    FaceDetector = None  # type: ignore[misc, assignment]

def _default_model_path() -> Path:
    return Path.home() / ".cache" / "onevoice" / "models" / _MODEL_NAME

def _ensure_model(path: Path) -> Path:
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("downloading MediaPipe face model to %s", path)
    urllib.request.urlretrieve(_MODEL_URL, path)  # noqa: S310 - trusted Google CDN
    return path

class MediaPipeFaceDetector:

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
        model_path: str | Path | None = None,
    ) -> None:
        if mp is None or FaceDetector is None:
            raise RuntimeError(
                "mediapipe is required for MediaPipeFaceDetector; "
                "pip install 'onevoice[perception]' (or pip install mediapipe)"
            )
        del model_selection
        resolved = _ensure_model(Path(model_path) if model_path else _default_model_path())
        options = FaceDetectorOptions(
            base_options=mp_base_options.BaseOptions(model_asset_path=str(resolved)),
            running_mode=mp_running_mode.VisionTaskRunningMode.IMAGE,
            min_detection_confidence=float(min_detection_confidence),
        )
        self._min_confidence = float(min_detection_confidence)
        self._detector: Any = FaceDetector.create_from_options(options)

    def detect(self, frame: Frame) -> list[Detection]:
        image = frame.data
        if image is None:
            return []
        try:
            rgb = _to_rgb(image)
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            mp_img = mp_image.Image(
                image_format=mp_image.ImageFormat.SRGB, data=rgb
            )
            results = self._detector.detect(mp_img)
        except Exception:  # noqa: BLE001 - detector must never crash the loop
            logger.exception("mediapipe detection failed; returning no faces")
            return []

        detections: list[Detection] = []
        for det in getattr(results, "detections", None) or []:
            parsed = self._parse(det, width, height)
            if parsed is not None:
                detections.append(parsed)
        return detections

    def _parse(self, det: Any, width: int, height: int) -> Detection | None:
        bbox = det.bounding_box
        x = float(bbox.origin_x)
        y = float(bbox.origin_y)
        w = float(bbox.width)
        h = float(bbox.height)
        if w <= 0 or h <= 0:
            return None

        landmarks: dict[str, tuple[float, float]] = {}
        for kp in det.keypoints or []:
            label = kp.label or "keypoint"
            if kp.x is not None and kp.y is not None:
                landmarks[label] = (float(kp.x) * width, float(kp.y) * height)

        mouth = (
            landmarks.get("mouth")
            or landmarks.get("mouth_center")
            or landmarks.get("mouthCenter")
            or landmarks.get("Mouth")
        )
        if mouth is not None:
            landmarks["mouth"] = mouth
        elif landmarks:

            mouth = (x + w * 0.5, y + h * 0.75)
            landmarks["mouth"] = mouth

        lip_bbox = None
        if mouth is not None:
            mx, my = mouth

            lw, lh = w * 0.65, h * 0.45
            top_bias = 0.6
            lip_bbox = (mx - lw / 2.0, my - lh * top_bias, lw, lh)

        score = self._min_confidence
        if det.categories:
            score = float(det.categories[0].score or score)

        return Detection(
            bounding_box=(x, y, w, h),
            confidence=score,
            landmarks=landmarks or None,
            lip_bbox=lip_bbox,
        )

    def close(self) -> None:
        detector = getattr(self, "_detector", None)
        if detector is not None:
            detector.close()

def _to_rgb(image: Any) -> Any:

    import numpy as np

    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr[:, :, ::-1].copy()
    return arr
