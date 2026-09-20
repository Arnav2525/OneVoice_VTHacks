

from __future__ import annotations

import hashlib
import logging
import tempfile
import threading
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.video.trackers.iou_tracker import iou

logger = logging.getLogger(__name__)
_MODEL_ROOT = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
_MODELS = {
    "face_detection_yunet/face_detection_yunet_2023mar.onnx": (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
    ),
    "face_recognition_sface/face_recognition_sface_2021dec.onnx": (
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
    ),
}

def _model_path(relative: str, cache: Path) -> Path:

    target = cache / Path(relative).name
    expected = _MODELS[relative]
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache, suffix=".download", delete=False
        ) as f:
            temporary = Path(f.name)
            with urllib.request.urlopen(
                f"{_MODEL_ROOT}/{relative}", timeout=30
            ) as response:
                while block := response.read(1024 * 1024):
                    f.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Face model checksum did not match; download discarded")
        temporary.replace(target)
        return target
    except Exception as exc:
        raise RuntimeError(
            "Local face matching could not load its models. Connect once to download "
            "the OpenCV weights, then retry Start listening."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

class LocalFaceEncoder:

    def __init__(self, cache: Path | None = None) -> None:
        cache = cache or Path.home() / ".cache" / "onevoice" / "models"
        detection, recognition = [_model_path(name, cache) for name in _MODELS]
        self.detector = cv2.FaceDetectorYN.create(
            str(detection),
            "",
            (320, 320),
            0.8,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(recognition),
            "",
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )

    def encode(self, frame: Any, tracks: list[SpeakerTrack]) -> dict[str, np.ndarray]:
        if frame.data is None or not tracks:
            return {}
        image = np.asarray(frame.data)
        if image.ndim != 3 or image.shape[2] != 3:
            return {}
        height, width = image.shape[:2]
        scale = min(1.0, 640.0 / max(height, width))
        small = cv2.resize(image, (round(width * scale), round(height * scale)))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        if faces is None:
            return {}

        faces = faces.copy()
        faces[:, :14] /= scale
        pairs = sorted(
            (
                (iou(track.bounding_box, tuple(face[:4])), ti, fi)
                for ti, track in enumerate(tracks)
                for fi, face in enumerate(faces)
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_faces: set[int] = set()
        result = {}
        for overlap, ti, fi in pairs:
            if overlap < 0.3 or ti in used_tracks or fi in used_faces:
                continue
            used_tracks.add(ti)
            used_faces.add(fi)
            if min(faces[fi][2:4]) < 32:
                continue
            aligned = self.recognizer.alignCrop(image, faces[fi])
            feature = _unit(self.recognizer.feature(aligned))
            if feature is not None:
                result[tracks[ti].track_id] = feature
        return result

def _unit(value: Any) -> np.ndarray | None:
    feature = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if not feature.size or not np.isfinite(feature).all() or norm < 1e-8:
        return None
    return feature / norm

@dataclass
class _Profile:
    anchor: np.ndarray
    recent: np.ndarray

@dataclass
class _Candidate:
    feature: np.ndarray
    hits: int
    last_seen_ms: float

class SessionFaceTracker:

    def __init__(
        self,
        tracker: Any,
        encoder: Any,
        *,
        max_people: int = 3,
        confirm_frames: int = 3,
        match_threshold: float = 0.5,
        novelty_threshold: float = 0.25,
        ambiguity_margin: float = 0.08,
        coast_ms: float = 750.0,
    ) -> None:
        self.tracker, self.encoder = tracker, encoder
        self.max_people, self.confirm_frames = max_people, confirm_frames
        self.match_threshold = match_threshold
        self.novelty_threshold = novelty_threshold
        self.ambiguity_margin = ambiguity_margin

        self.coast_ms = coast_ms
        self._profiles: list[_Profile] = []
        self._assignments: dict[str, tuple[int, float]] = {}
        self._pending: dict[str, _Candidate] = {}
        self._last_frame_ms: float | None = None
        self._last_output: list[SpeakerTrack] = []
        self._status_lock = threading.Lock()
        self._status: dict[str, Any] = {}

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def process_frame(self, frame: Any) -> list[SpeakerTrack]:

        if self._last_frame_ms == frame.timestamp_ms:
            return list(self._last_output)
        tracks = sorted(
            (
                t
                for t in self.tracker.process_frame(frame)
                if t.metadata.get("visible", True)
            ),
            key=lambda t: t.bounding_box[0],
        )
        features: dict[str, np.ndarray] = {}
        error = False
        try:
            for key, value in self.encoder.encode(frame, tracks).items():
                feature = _unit(value)
                if feature is not None:
                    features[key] = feature
        except Exception:
            logger.exception("Local face matching failed; withholding person labels")
            features.clear()
            error = True
        now = frame.timestamp_ms
        self._pending = {
            key: value
            for key, value in self._pending.items()
            if now - value.last_seen_ms <= 1000 and key in features
        }
        observed = [(t, features[t.track_id]) for t in tracks if t.track_id in features]
        scores = np.array(
            [
                [max(float(f @ p.anchor), float(f @ p.recent)) for p in self._profiles]
                for _, f in observed
            ],
            dtype=np.float32,
        ).reshape(len(observed), len(self._profiles))
        assigned: dict[int, int] = {}
        novel = []
        for row, (track, feature) in enumerate(observed):
            ranked = np.argsort(scores[row])[::-1]
            best = float(scores[row, ranked[0]]) if len(ranked) else -1.0
            if best < self.novelty_threshold:
                novel.append(row)
                continue
            self._pending.pop(track.track_id, None)
            if best < self.match_threshold:
                continue
            person = int(ranked[0])
            runner_up = float(scores[row, ranked[1]]) if len(ranked) > 1 else -1.0
            competitors = np.delete(scores[:, person], row)
            if best - runner_up < self.ambiguity_margin or (
                competitors.size
                and best - float(competitors.max()) < self.ambiguity_margin
            ):
                continue
            assigned[row] = person
            profile = self._profiles[person]
            if best >= 0.65 and float(feature @ profile.anchor) >= self.match_threshold:
                profile.recent = _unit(0.9 * profile.recent + 0.1 * feature)

        for row in novel:
            track, feature = observed[row]
            if len(self._profiles) >= self.max_people:
                continue

            if any(
                float(feature @ p.anchor) >= self.novelty_threshold
                for p in self._profiles
            ):
                continue
            previous = self._pending.get(track.track_id)
            hits = (
                previous.hits + 1
                if (
                    previous is not None
                    and float(feature @ previous.feature) >= self.match_threshold
                )
                else 1
            )
            self._pending[track.track_id] = _Candidate(feature.copy(), hits, now)
            if hits >= self.confirm_frames:
                assigned[row] = len(self._profiles)
                self._profiles.append(_Profile(feature.copy(), feature.copy()))
                del self._pending[track.track_id]

        def _label(track, person, coasted):
            return replace(
                track,
                track_id=f"person-{person + 1}",
                metadata={
                    **track.metadata,
                    "person_number": person + 1,
                    "identity_matching": "local",
                    "raw_track_id": track.track_id,
                    "identity_coasted": coasted,
                },
            )

        output = []
        claimed: set[int] = set()
        confident: set[str] = set()
        for row, person in assigned.items():
            track = observed[row][0]
            self._assignments[track.track_id] = (person, now)
            confident.add(track.track_id)
            claimed.add(person)
            output.append(_label(track, person, False))

        self._assignments = {
            key: value
            for key, value in self._assignments.items()
            if now - value[1] <= self.coast_ms
        }

        coasted = 0
        for track in tracks:
            if track.track_id in confident:
                continue
            entry = self._assignments.get(track.track_id)
            if entry is None:
                continue
            person, _last_ms = entry
            if person in claimed:
                continue
            feature = features.get(track.track_id)
            if feature is not None:
                profile = self._profiles[person]
                resemblance = max(
                    float(feature @ profile.anchor), float(feature @ profile.recent)
                )
                if resemblance < self.novelty_threshold:

                    self._assignments.pop(track.track_id, None)
                    continue
            claimed.add(person)
            coasted += 1
            output.append(_label(track, person, True))

        with self._status_lock:
            self._status = {
                "enabled": True,
                "registered": len(self._profiles),
                "capacity": self.max_people,
                "unmatched": len(tracks) - len(output),
                "coasted": coasted,
                "error": error,
            }
        self._last_frame_ms = frame.timestamp_ms
        self._last_output = sorted(output, key=lambda t: t.metadata["person_number"])
        return list(self._last_output)

    def close(self) -> None:
        self._profiles.clear()
        self._assignments.clear()
        self._pending.clear()
        self._last_frame_ms = None
        self._last_output.clear()
        with self._status_lock:
            self._status = {}
        close = getattr(self.tracker, "close", None)
        if callable(close):
            close()
