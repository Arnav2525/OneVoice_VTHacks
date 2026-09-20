

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.video.detectors.base import Detection, FaceDetector

logger = logging.getLogger(__name__)

try:
    from onevoice.video.asd.motion_scorer import MotionActiveSpeakerScorer
except ImportError:  # pragma: no cover
    MotionActiveSpeakerScorer = None  # type: ignore[misc, assignment]

Box = tuple[float, float, float, float]

@dataclass(frozen=True)
class TrackerConfig:

    iou_threshold: float = 0.3
    min_hits: int = 3
    max_age: int = 30
    reid_window_ms: float = 2500.0
    reid_iou_threshold: float = 0.2
    emit_coasting: bool = False
    coast_max_misses: int = 5
    id_prefix: str = "face"

    attach_frame_bgr: bool = False

@dataclass
class _Track:
    track_id: str
    box: Box
    confidence: float
    velocity: tuple[float, float]
    hits: int
    misses: int
    age: int
    last_seen_ms: float
    detection: Detection | None
    confirmed: bool = False

@dataclass
class _LostTrack:
    track: _Track
    lost_since_ms: float

def _center(box: Box) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)

def iou(a: Box, b: Box) -> float:

    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0.0 else 0.0

class IouFaceTracker:

    def __init__(
        self,
        detector: FaceDetector,
        config: TrackerConfig | None = None,
        asd_scorer: Any | None = None,
    ) -> None:
        self._detector = detector
        self._config = config or TrackerConfig()
        if asd_scorer is not None:
            self._asd = asd_scorer
        elif MotionActiveSpeakerScorer is not None:
            self._asd = MotionActiveSpeakerScorer()
        else:
            self._asd = None
        self._tracks: list[_Track] = []
        self._lost: list[_LostTrack] = []
        self._next_id = 0
        self._current_frame: Frame | None = None
        self._stats_lock = threading.Lock()
        self._stats: dict[str, Any] = {
            "frames_processed": 0,
            "tracks_created": 0,
            "reid_recoveries": 0,
            "active_tracks": 0,
            "lost_tracks": 0,
        }

    def process_frame(self, frame: Frame) -> list[SpeakerTrack]:
        self._current_frame = frame
        now = frame.timestamp_ms
        try:
            detections = self._detector.detect(frame)
        except Exception:  # noqa: BLE001 - detector must never crash the loop
            logger.exception("detector failed; coasting existing tracks")
            detections = []

        self._predict()
        matched_tracks, unmatched_dets = self._associate(detections)
        self._age_unmatched(matched_tracks)
        self._update_matched(matched_tracks, now)

        remaining = self._reid(unmatched_dets, now)
        for det in remaining:
            self._spawn(det, now)

        self._prune(now)
        output = self._emit(now)
        if self._asd is not None:
            scored: list[SpeakerTrack] = []
            for track in output:
                score = self._asd.score(track)
                meta = dict(track.metadata)
                meta["speaking_score"] = score
                scored.append(
                    SpeakerTrack(
                        track_id=track.track_id,
                        bounding_box=track.bounding_box,
                        confidence=track.confidence,
                        metadata=meta,
                    )
                )
            output = scored
        self._record(len(output))
        return output

    def _predict(self) -> None:
        for t in self._tracks:
            if t.misses == 0:
                continue
            vx, vy = t.velocity
            x, y, w, h = t.box
            t.box = (x + vx, y + vy, w, h)

    def _associate(
        self, detections: list[Detection]
    ) -> tuple[dict[int, Detection], list[Detection]]:
        pairs: list[tuple[float, int, int]] = []
        for ti, t in enumerate(self._tracks):
            for di, det in enumerate(detections):
                score = iou(t.box, det.bounding_box)
                if score >= self._config.iou_threshold:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        matched_tracks: dict[int, Detection] = {}
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            matched_tracks[ti] = detections[di]
        unmatched = [d for i, d in enumerate(detections) if i not in used_dets]
        return matched_tracks, unmatched

    def _age_unmatched(self, matched_tracks: dict[int, Detection]) -> None:
        for ti, t in enumerate(self._tracks):
            if ti not in matched_tracks:
                t.misses += 1
                t.age += 1

    def _update_matched(
        self, matched_tracks: dict[int, Detection], now: float
    ) -> None:
        cfg = self._config
        for ti, det in matched_tracks.items():
            t = self._tracks[ti]
            old_cx, old_cy = _center(t.box)
            new_cx, new_cy = _center(det.bounding_box)
            if t.misses == 0:
                t.velocity = (new_cx - old_cx, new_cy - old_cy)
            t.box = det.bounding_box
            t.confidence = det.confidence
            t.detection = det
            t.hits += 1
            t.misses = 0
            t.age += 1
            t.last_seen_ms = now
            if t.hits >= cfg.min_hits:
                t.confirmed = True

    def _reid(self, detections: list[Detection], now: float) -> list[Detection]:
        if not self._lost or not detections:
            return detections
        cfg = self._config
        remaining: list[Detection] = []
        for det in detections:
            best_i = -1
            best_iou = cfg.reid_iou_threshold
            for i, lost in enumerate(self._lost):
                score = iou(lost.track.box, det.bounding_box)
                if score >= best_iou:
                    best_iou = score
                    best_i = i
            if best_i < 0:
                remaining.append(det)
                continue
            revived = self._lost.pop(best_i).track
            revived.box = det.bounding_box
            revived.confidence = det.confidence
            revived.detection = det
            revived.velocity = (0.0, 0.0)
            revived.hits += 1
            revived.misses = 0
            revived.last_seen_ms = now
            self._tracks.append(revived)
            with self._stats_lock:
                self._stats["reid_recoveries"] += 1
        return remaining

    def _spawn(self, det: Detection, now: float) -> None:
        cfg = self._config
        track_id = f"{cfg.id_prefix}-{self._next_id}"
        self._next_id += 1
        self._tracks.append(
            _Track(
                track_id=track_id,
                box=det.bounding_box,
                confidence=det.confidence,
                velocity=(0.0, 0.0),
                hits=1,
                misses=0,
                age=1,
                last_seen_ms=now,
                detection=det,
                confirmed=cfg.min_hits <= 1,
            )
        )
        with self._stats_lock:
            self._stats["tracks_created"] += 1

    def _prune(self, now: float) -> None:
        cfg = self._config
        still_active: list[_Track] = []
        for t in self._tracks:
            if t.misses > cfg.max_age:
                self._lost.append(_LostTrack(track=t, lost_since_ms=now))
            else:
                still_active.append(t)
        self._tracks = still_active
        self._lost = [
            lost
            for lost in self._lost
            if now - lost.lost_since_ms <= cfg.reid_window_ms
        ]

    def _emit(self, now: float) -> list[SpeakerTrack]:
        cfg = self._config
        frame_bgr = None
        if cfg.attach_frame_bgr and self._current_frame is not None:
            try:
                frame_bgr = np.asarray(self._current_frame.data).copy()
            except Exception:  # noqa: BLE001
                frame_bgr = None
        out: list[SpeakerTrack] = []
        for t in self._tracks:
            if not t.confirmed:
                continue
            visible = t.misses == 0
            if not visible and not (
                cfg.emit_coasting and t.misses <= cfg.coast_max_misses
            ):
                continue
            out.append(self._to_speaker_track(t, visible, frame_bgr=frame_bgr))
        return out

    def _to_speaker_track(
        self, t: _Track, visible: bool, *, frame_bgr: np.ndarray | None = None
    ) -> SpeakerTrack:
        metadata: dict[str, Any] = {
            "tracker": "iou",
            "det_confidence": t.confidence,
            "last_seen_ms": t.last_seen_ms,
            "age": t.age,
            "hits": t.hits,
            "misses": t.misses,
            "visible": visible,
        }
        if frame_bgr is not None:
            metadata["frame_bgr"] = frame_bgr
        det = t.detection
        if det is not None:
            if det.landmarks is not None:
                metadata["landmarks"] = det.landmarks
            if det.lip_bbox is not None:
                metadata["lip_roi"] = det.lip_bbox
                metadata["lip_roi_ts_ms"] = t.last_seen_ms
                patch = _crop_lip_patch(self._current_frame, det.lip_bbox)
                if patch is not None:
                    metadata["lip_patch"] = patch
        return SpeakerTrack(
            track_id=t.track_id,
            bounding_box=t.box,
            confidence=max(0.0, min(1.0, t.confidence)),
            metadata=metadata,
        )

    def _record(self, emitted: int) -> None:
        with self._stats_lock:
            self._stats["frames_processed"] += 1
            self._stats["active_tracks"] = len(self._tracks)
            self._stats["lost_tracks"] = len(self._lost)
            self._stats["emitted_last_frame"] = emitted

    def get_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return dict(self._stats)

    def close(self) -> None:

        close = getattr(self._detector, "close", None)
        if callable(close):
            close()
        self._tracks.clear()
        self._lost.clear()
        self._current_frame = None

LIP_PATCH_SIZE = 88

_LRW_MEAN = 0.421
_LRW_STD = 0.165

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None  # type: ignore[assignment,misc]

def _resize_patch(patch: np.ndarray, size: int) -> np.ndarray:

    if patch.shape == (size, size):
        return patch
    if cv2 is not None:
        return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)

    src_h, src_w = patch.shape
    row_idx = (np.arange(size) * src_h / size).astype(np.int64).clip(0, src_h - 1)
    col_idx = (np.arange(size) * src_w / size).astype(np.int64).clip(0, src_w - 1)
    return patch[row_idx][:, col_idx]

def _crop_lip_patch(
    frame: Frame | None, lip_bbox: Box
) -> np.ndarray | None:

    if frame is None or frame.data is None:
        return None
    try:
        image = np.asarray(frame.data)
    except Exception:  # noqa: BLE001
        return None
    if image.ndim < 2:
        return None
    h, w = int(image.shape[0]), int(image.shape[1])
    x, y, bw, bh = lip_bbox
    x0 = int(round(x))
    y0 = int(round(y))
    x1 = int(round(x + bw))
    y1 = int(round(y + bh))
    oh, ow = y1 - y0, x1 - x0
    if oh <= 0 or ow <= 0:
        return None
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    src = image[sy0:sy1, sx0:sx1]
    if src.ndim == 3:
        src = src.mean(axis=2)
    canvas = np.zeros((oh, ow), dtype=np.float32)
    canvas[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = src.astype(np.float32)
    patch = canvas
    if patch.size == 0:
        return None
    patch = _resize_patch(patch, LIP_PATCH_SIZE)

    patch = (patch / 255.0 - _LRW_MEAN) / _LRW_STD
    return patch
