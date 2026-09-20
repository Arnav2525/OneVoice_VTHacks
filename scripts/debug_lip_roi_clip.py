

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2
import numpy as np

from onevoice.core.models.frame import Frame
from onevoice.video.detectors.mediapipe_detector import MediaPipeFaceDetector
from onevoice.video.trackers.iou_tracker import IouFaceTracker, TrackerConfig

MOUTH_FPS = 25
WINDOW_FRAMES = int(1.5 * MOUTH_FPS)

def main() -> None:
    clip_id, target, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = REPO_ROOT / "data" / "dolphin_tier_a" / "mouth_cache" / clip_id / target / "temp_25fps.mp4"
    cap = cv2.VideoCapture(str(video_path))
    frames_bgr = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()
    print(f"{clip_id}/{target}: {len(frames_bgr)} frames, video={video_path}")

    detector = MediaPipeFaceDetector()
    tracker = IouFaceTracker(detector, TrackerConfig(min_hits=1))

    n = len(frames_bgr)

    found_windows = []
    for start in range(0, max(1, n - WINDOW_FRAMES + 1), 5):
        probe_tracker = IouFaceTracker(detector, TrackerConfig(min_hits=1))
        lip_bbox = None
        face_bbox = None
        for i in range(start, min(start + WINDOW_FRAMES, n)):
            frame_bgr = frames_bgr[i]
            h, w = frame_bgr.shape[:2]

            frame = Frame(
                timestamp_ms=float(i * 1000.0 / MOUTH_FPS),
                data=frame_bgr,
                metadata={"width": w, "height": h},
            )
            tracks = probe_tracker.process_frame(frame)
            if tracks and "lip_roi" in tracks[0].metadata:
                lip_bbox = tracks[0].metadata["lip_roi"]
                face_bbox = tracks[0].bounding_box
                break
        found_windows.append((start, lip_bbox, face_bbox))
        if lip_bbox is not None:
            break

    for start, lip_bbox, face_bbox in found_windows:
        status = "FOUND" if lip_bbox else "no-track-yet"
        print(f"  window start={start} ({start/MOUTH_FPS:.2f}s): {status} lip_bbox={lip_bbox} face_bbox={face_bbox}")

    if not found_windows or found_windows[-1][1] is None:
        print("No face window found at all.")
        detector.close()
        return

    start_frame, lip_bbox, face_bbox = found_windows[-1]
    print(f"\nUsing window start_frame={start_frame} ({start_frame/MOUTH_FPS:.2f}s)")
    print(f"lip_bbox = {lip_bbox}")
    print(f"face_bbox = {face_bbox}")

    idxs = sorted(set([
        start_frame,
        start_frame + WINDOW_FRAMES // 4,
        start_frame + WINDOW_FRAMES // 2,
        start_frame + 3 * WINDOW_FRAMES // 4,
        min(start_frame + WINDOW_FRAMES - 1, n - 1),
    ]))
    for idx in idxs:
        frame_bgr = frames_bgr[idx].copy()
        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            cv2.rectangle(frame_bgr, (int(fx), int(fy)), (int(fx + fw), int(fy + fh)), (0, 255, 0), 1)
        if lip_bbox is not None:
            lx, ly, lw, lh = lip_bbox
            cv2.rectangle(frame_bgr, (int(lx), int(ly)), (int(lx + lw), int(ly + lh)), (0, 0, 255), 1)
        out_path = out_dir / f"{clip_id}_{target}_frame{idx:04d}.png"
        cv2.imwrite(str(out_path), frame_bgr)
        print(f"  saved {out_path}")

        if lip_bbox is not None:
            h, w = frame_bgr.shape[:2]
            x0, y0 = max(0, int(lx)), max(0, int(ly))
            x1, y1 = min(w, int(lx + lw)), min(h, int(ly + lh))
            crop = frames_bgr[idx][y0:y1, x0:x1]
            if crop.size > 0:
                crop_big = cv2.resize(crop, (crop.shape[1] * 6, crop.shape[0] * 6), interpolation=cv2.INTER_NEAREST)
                crop_path = out_dir / f"{clip_id}_{target}_frame{idx:04d}_lipcrop.png"
                cv2.imwrite(str(crop_path), crop_big)
                print(f"  saved {crop_path}")

    detector.close()

if __name__ == "__main__":
    main()
