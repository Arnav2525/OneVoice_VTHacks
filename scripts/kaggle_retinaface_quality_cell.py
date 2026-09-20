

import os
import subprocess
import sys
from pathlib import Path

repo = Path.cwd()
script_candidates = [
    repo / "scripts" / "verify_retinaface_alignment.py",
    repo / "verify_retinaface_alignment.py",
]
script = next((p for p in script_candidates if p.is_file()), None)
if script is None:
    raise FileNotFoundError(
        "verify_retinaface_alignment.py not in this session's code copy.\n"
        f"Looked in: {[str(p) for p in script_candidates]}\n"
        "Re-package + re-upload onevoice-kaggle-code.zip (package_kaggle_bundle.ps1), "
        "then re-run Section 1 (copy code into /kaggle/working/onevoice)."
    )

mouth_candidates = [
    Path(DATASET_TT_DIR).resolve().parent / "mouth_cache",
    Path("/kaggle/working/dolphin_tier_a/mouth_cache"),
    Path(DATASET_TT_DIR).resolve().parent.parent / "mouth_cache",
]
mouth = next((p for p in mouth_candidates if p.is_dir()), None)
if mouth is None:
    raise FileNotFoundError(
        "mouth_cache/ not found next to the tier-a dataset.\n"
        f"Looked in: {[str(p) for p in mouth_candidates]}\n"
        "Re-package + re-upload onevoice-dolphin-tier-a.zip (now includes mouth_cache/), "
        "or attach a dataset that has mouth_cache/ beside tt/."
    )

cmd = [
    sys.executable,
    str(script),
    "--device",
    "cuda",
    "--tt-root",
    TEST_DIR,
    "--mouth-cache-root",
    str(mouth),
]

print("script =", script)
print("mouth_cache =", mouth)
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout[-10000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
print("RETURN CODE:", result.returncode)
