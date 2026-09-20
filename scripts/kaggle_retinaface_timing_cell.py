

import subprocess
import sys

RETINAFACE_TIMING_VIDEO = f"{DATASET_TT_DIR}/../mouth_cache/corpus_mix_01/s1/temp_25fps.mp4"

cmd = [
    sys.executable,
    "scripts/time_retinaface_gpu.py",
    "--video",
    RETINAFACE_TIMING_VIDEO,
    "--n-frames",
    "50",
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout[-6000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
print("RETURN CODE:", result.returncode)
