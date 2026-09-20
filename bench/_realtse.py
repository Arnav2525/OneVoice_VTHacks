

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("onevoice.bench.realtse")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_VENDOR = _REPO_ROOT / "vendor" / "realtse"
_DEFAULT_CHECKPOINTS = _REPO_ROOT / "vendor" / "realtse_checkpoints"
SAMPLE_RATE = 16_000

VARIANTS = (
    "spk_emb_100",
    "spk_emb_causal_100",
    "tfmap_context_100",
    "tfmap_context_causal_100",
)

def vendor_root() -> Path:

    root = Path(os.environ.get("REALTSE_VENDOR_ROOT", _DEFAULT_VENDOR))
    if not (root / "wesep").is_dir():
        raise RuntimeError(
            f"REAL-TSE vendor tree missing at {root}. "
            "Run: python scripts/vendor_realtse.py"
        )
    return root.resolve()

def ensure_realtse_import_path() -> Path:
    root = vendor_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root

def checkpoint_root() -> Path:

    root = Path(os.environ.get("REALTSE_CHECKPOINT_ROOT", _DEFAULT_CHECKPOINTS))
    if not root.is_dir():
        raise RuntimeError(
            f"REAL-TSE checkpoints missing at {root}. "
            "Run: python scripts/fetch_realtse_checkpoints.py"
        )
    return root.resolve()

def _stub_unused_wespeaker_frontends() -> None:

    import types

    for mod_name, cls_name in (
        ("wespeaker.frontend.s3prl", "S3prlFrontend"),
        ("wespeaker.frontend.whisper_encoder", "whisper_encoder"),
        ("wespeaker.frontend.w2vbert", "W2VBertFrontend"),
    ):
        if mod_name in sys.modules:
            continue
        stub = types.ModuleType(mod_name)
        setattr(stub, cls_name, object)
        sys.modules[mod_name] = stub

@dataclass(frozen=True)
class RealTSELoadStats:
    variant: str
    device: str
    load_ms: float
    param_count: int | None
    causal: bool

def load_realtse(
    variant: str = "spk_emb_causal_100",
    *,
    device: str | None = None,
) -> tuple[Any, str, RealTSELoadStats]:

    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")

    ensure_realtse_import_path()
    import torch

    _stub_unused_wespeaker_frontends()
    import wesep

    pretrain_dir = checkpoint_root() / variant
    if not (pretrain_dir / "avg_model.pt").is_file():
        raise RuntimeError(f"missing avg_model.pt under {pretrain_dir}")

    if device is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "requested --device cuda but torch.cuda.is_available() is False"
        )

    t0 = time.perf_counter()
    model = wesep.load_model_local(str(pretrain_dir))
    model.set_device(dev)
    model.set_resample_rate(SAMPLE_RATE)
    load_ms = (time.perf_counter() - t0) * 1000.0

    param_count: int | None = None
    net = getattr(model, "model", None) or getattr(model, "net", None)
    if net is not None and hasattr(net, "parameters"):
        param_count = sum(p.numel() for p in net.parameters())

    stats = RealTSELoadStats(
        variant=variant,
        device=dev,
        load_ms=load_ms,
        param_count=param_count,
        causal="causal" in variant,
    )
    logger.info(
        "loaded REAL-TSE %s on %s (%.0f ms, %s params, causal=%s)",
        variant,
        dev,
        load_ms,
        param_count,
        stats.causal,
    )
    return model, dev, stats

def infer_separation(model: Any, mixture_wav: str | Path, enroll_wav: str | Path) -> Any:

    import numpy as np

    est = model.extract_speech(str(mixture_wav), str(enroll_wav))
    if est is None:
        raise RuntimeError(f"REAL-TSE returned no output for {mixture_wav}")
    arr = est[0] if hasattr(est, "__getitem__") and not np.isscalar(est) else est
    out = np.asarray(arr, dtype=np.float64)
    return out.reshape(-1)

def eval_manifest(
    test_dir: str | Path,
    *,
    segment_seconds: float = 2.0,
    enroll_seconds: float = 3.0,
    enroll_offset_seconds: float | None = None,
) -> list[dict[str, Any]]:

    root = Path(test_dir)
    with open(root / "mix.json", encoding="utf-8") as f:
        mix_infos = json.load(f)
    with open(root / "s1.json", encoding="utf-8") as f:
        s1_infos = json.load(f)
    with open(root / "s2.json", encoding="utf-8") as f:
        s2_infos = json.load(f)

    seg_len = int(segment_seconds * SAMPLE_RATE)
    enroll_len = int(enroll_seconds * SAMPLE_RATE)
    enroll_offset = (
        seg_len
        if enroll_offset_seconds is None
        else int(enroll_offset_seconds * SAMPLE_RATE)
    )

    items: list[dict[str, Any]] = []
    for i in range(len(mix_infos) - 1, -1, -1):
        n_samples = int(mix_infos[i][1])
        if n_samples < seg_len:
            continue
        mix_path = Path(mix_infos[i][0])
        clip = mix_path.parent.name
        for target, src_info in (("s1", s1_infos[i]), ("s2", s2_infos[i])):
            ref_path = Path(src_info[0])
            disjoint = n_samples >= enroll_offset + enroll_len
            items.append(
                {
                    "clip_id": f"{clip}/{target}",
                    "target": target,
                    "mix_path": str(mix_path),
                    "ref_path": str(ref_path),
                    "n_samples": n_samples,
                    "seg_len": seg_len,
                    "enroll_offset": enroll_offset if disjoint else 0,
                    "enroll_len": enroll_len if disjoint else min(enroll_len, n_samples),
                    "enroll_disjoint": disjoint,
                }
            )
    return items

def write_segment_wav(
    src_wav: str | Path, dest_wav: str | Path, *, offset: int, length: int
) -> Path:

    import soundfile as sf

    audio, sr = sf.read(str(src_wav), dtype="float32")
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"expected {SAMPLE_RATE} Hz, got {sr} Hz in {src_wav}")
    segment = audio[offset : offset + length]
    if len(segment) < length:
        raise RuntimeError(
            f"requested [{offset}:{offset + length}) but {src_wav} has only "
            f"{len(audio)} samples"
        )
    dest = Path(dest_wav)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), segment, SAMPLE_RATE)
    return dest
