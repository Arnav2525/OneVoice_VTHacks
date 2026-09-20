
import sys
sys.path.insert(0, "src")

import cv2

from onevoice.core.models.frame import Frame
from onevoice.video.detectors.mediapipe_detector import MediaPipeFaceDetector
from onevoice.video.trackers.iou_tracker import IouFaceTracker, TrackerConfig

path = "data/dolphin_tier_a/mouth_cache/corpus_mix_01/s1/temp_25fps.mp4"
cap = cv2.VideoCapture(path)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print(f"read {len(frames)} frames from {path}")

detector = MediaPipeFaceDetector()
tracker = IouFaceTracker(detector, TrackerConfig(min_hits=3))

scores = []
for i, frame_bgr in enumerate(frames):
    h, w = frame_bgr.shape[:2]
    frame = Frame(timestamp_ms=float(i * 1000.0 / 25.0), data=frame_bgr, metadata={"width": w, "height": h})
    tracks = tracker.process_frame(frame)
    if not tracks:
        scores.append(None)
        continue
    t = tracks[0]
    scores.append(t.metadata.get("speaking_score"))

print("frame: score (None = no confirmed track)")
for i, s in enumerate(scores):
    print(f"{i:3d}: {s}")

real = [s for s in scores if s is not None]
print(f"\nconfirmed-track frames: {len(real)}/{len(scores)}")
if real:
    print(f"max score: {max(real):.3f}  mean score: {sum(real)/len(real):.3f}")
    above = sum(1 for s in real if s >= 0.55)
    print(f"frames >= acquire_thresh=0.55: {above}/{len(real)}")
