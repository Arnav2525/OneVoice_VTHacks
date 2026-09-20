

import json
import subprocess
import sys
from pathlib import Path

cmd = [
    sys.executable,
    "-m",
    "bench.bench_dolphin_offline",
    "--test-dir",
    TEST_DIR,
    "--segment-seconds",
    str(SEGMENT_SECONDS),
    "--max-clips",
    str(MAX_CLIPS),
    "--device",
    "cuda",
    "--dolphin-light",
    "--window-ms",
    str(WINDOW_MS),
    "--output-buffer-ms",
    str(OUTPUT_BUFFER_MS),
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout[-6000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
print("RETURN CODE:", result.returncode)
