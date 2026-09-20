

import subprocess
import sys

CONTENTION_TEST_VIDEO = f"{DATASET_TT_DIR}/../mouth_cache/corpus_mix_01/s1/temp_25fps.mp4"

cmd = [
    sys.executable,
    "scripts/measure_gpu_contention.py",
    "--test-dir",
    TEST_DIR,
    "--video",
    CONTENTION_TEST_VIDEO,
    "--n-dolphin",
    "6",
    "--n-retinaface",
    "40",
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout[-6000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
print("RETURN CODE:", result.returncode)
