

import os
import subprocess
import sys

repo = "/kaggle/working/onevoice"
os.chdir(repo)

vendor_dest = os.path.join(repo, "vendor", "realtse")
ckpt_dest = os.path.join(repo, "vendor", "realtse_checkpoints")

subprocess.run(
    [sys.executable, "vendor_realtse.py", "--dest", vendor_dest], check=True
)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
vendor_req = os.path.join(vendor_dest, "requirements.txt")
if os.path.isfile(vendor_req):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", vendor_req], check=True
    )
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/wenet-e2e/wespeaker.git@8f53b6485d9f88a207bd17e7f8dba899495ec794",
    ],
    check=True,
)

subprocess.run(
    [sys.executable, "fetch_realtse_checkpoints.py", "--dest", ckpt_dest], check=True
)

cmd = [
    sys.executable, "-m", "bench.bench_realtse_offline",
    "--test-dir", "/kaggle/working/dolphin_tier_a/tt",
    "--variant", "spk_emb_causal_100",
    "--segment-seconds", "2.0",
    "--enroll-seconds", "3.0",
    "--max-clips", "8",
    "--device", "cuda",
    "--window-ms", "2000.0",
    "--output-buffer-ms", "100.0",
    "--output-dir", "/kaggle/working/runs",
]
proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout)
if proc.returncode != 0:
    print(proc.stderr)
    raise RuntimeError("realtse bench failed")
