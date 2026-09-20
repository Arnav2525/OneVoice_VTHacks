

import subprocess
import sys

try:
    import mediapipe  # noqa: F401
except ImportError:
    print("mediapipe missing (perception extra, not in Section 1's dolphin,bench install) -- installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe"], check=True)

LIPROI_MOUTH_CACHE_ROOT = f"{DATASET_TT_DIR}/../mouth_cache"

cmd = [
    sys.executable,
    "scripts/verify_lip_roi_real_crop.py",
    "--device",
    "cuda",
    "--window-s",
    "2.0",
    "--tt-root",
    TEST_DIR,
    "--mouth-cache-root",
    LIPROI_MOUTH_CACHE_ROOT,
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout[-10000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
print("RETURN CODE:", result.returncode)
