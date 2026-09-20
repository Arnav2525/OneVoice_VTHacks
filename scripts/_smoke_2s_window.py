import sys
sys.path.insert(0, "src")
sys.path.insert(0, ".")

import verify_lip_roi_real_crop as m
from bench._dolphin import load_dolphin
from onevoice.video.detectors.mediapipe_detector import MediaPipeFaceDetector

m.WINDOW_S = 2.0
m.WINDOW_SAMPLES = int(m.WINDOW_S * m.SAMPLE_RATE)
m.WINDOW_FRAMES = int(m.WINDOW_S * m.MOUTH_FPS)
print(f"window_s={m.WINDOW_S} samples={m.WINDOW_SAMPLES} frames={m.WINDOW_FRAMES}")

print("loading dolphin...")
model, device, stats = load_dolphin(device="cpu", warmup=False)
_detector = MediaPipeFaceDetector()

for clip_id, target in [("corpus_mix_01", "s1"), ("corpus_mix_04", "s1")]:
    print(f"\n{clip_id}/{target}...")
    result = m.process_clip(clip_id, target, model, device, lambda: _detector)
    print("  ", result)
print("\ndone")
