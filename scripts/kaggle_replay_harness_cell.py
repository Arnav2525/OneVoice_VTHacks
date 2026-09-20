

import subprocess
import sys
from pathlib import Path

from IPython.display import Audio, display

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".[perception]"],
    check=False,
)

_tt = Path(globals().get("DATASET_TT_DIR", "/kaggle/input/onevoice-dolphin-tier-a/tt"))
_test = Path(globals().get("TEST_DIR", "/kaggle/working/dolphin_tier_a/tt"))
_data_root = _tt.parent

candidates = [
    _data_root / "mouth_cache" / "corpus_mix_01" / "s1" / "temp_25fps.mp4",
    Path("/kaggle/input/onevoice-dolphin-tier-a/mouth_cache/corpus_mix_01/s1/temp_25fps.mp4"),
]
REPLAY_VIDEO = next((str(p) for p in candidates if p.is_file()), None)
REPLAY_AUDIO = str(_test / "corpus_mix_01" / "mix.wav")
if not Path(REPLAY_AUDIO).is_file():
    REPLAY_AUDIO = str(_tt / "corpus_mix_01" / "mix.wav")
REPLAY_OUTPUT_DIR = "/kaggle/working/replay_kaggle_gpu"
REPLAY_CONFIG = "configs/experiments/dolphin_live.yaml"

missing = []
if REPLAY_VIDEO is None:
    missing.append("mouth_cache/.../temp_25fps.mp4 (re-upload data zip with mouth_cache/)")
if not Path(REPLAY_AUDIO).is_file():
    missing.append(REPLAY_AUDIO)
if not Path("scripts/replay_harness.py").is_file():
    missing.append("scripts/replay_harness.py (re-package + re-upload code zip)")
if not Path(REPLAY_CONFIG).is_file():
    missing.append(f"{REPLAY_CONFIG} (re-package + re-upload code zip)")
if missing:
    raise FileNotFoundError(
        "Replay harness prerequisites missing:\n  - "
        + "\n  - ".join(missing)
        + "\nRe-run: powershell -File scripts/package_kaggle_bundle.ps1"
    )

cmd = [
    sys.executable,
    "scripts/replay_harness.py",
    "--video",
    REPLAY_VIDEO,
    "--audio",
    REPLAY_AUDIO,
    "--config",
    REPLAY_CONFIG,
    "--output-dir",
    REPLAY_OUTPUT_DIR,
    "--log-level",
    "INFO",
]

print("Running:", " ".join(cmd))
print(f"config: {REPLAY_CONFIG}")
print(f"video: {REPLAY_VIDEO}")
print(f"audio: {REPLAY_AUDIO}")
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout[-8000:])

print("STDERR:", result.stderr[-6000:])
print("RETURN CODE:", result.returncode)

mixture_path = Path(REPLAY_OUTPUT_DIR) / "mixture_before.wav"
separated_path = Path(REPLAY_OUTPUT_DIR) / "separated_after.wav"

if mixture_path.is_file():
    print("\nInput mixture (before separation):")
    display(Audio(str(mixture_path)))
else:
    print(f"\nmissing {mixture_path} -- check STDERR above for what went wrong.")

if separated_path.is_file():
    print("\nSeparated output (after Dolphin, real-time-paced on GPU):")
    display(Audio(str(separated_path)))
else:
    print(f"\nmissing {separated_path} -- check STDERR above for what went wrong.")
