

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("onevoice.bench.dolphin")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_VENDOR = _REPO_ROOT / "vendor" / "dolphin"
SAMPLE_RATE = 16_000
MOUTH_FPS = 25

def vendor_root() -> Path:

    root = Path(os.environ.get("DOLPHIN_VENDOR_ROOT", _DEFAULT_VENDOR))
    if not (root / "look2hear").is_dir():
        raise RuntimeError(
            f"Dolphin vendor tree missing at {root}. "
            "Run: python scripts/vendor_dolphin.py"
        )
    return root.resolve()

def ensure_dolphin_import_path() -> Path:
    root = vendor_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root

@dataclass(frozen=True)
class DolphinLoadStats:
    model_id: str
    device: str
    load_ms: float
    warmup_ms: float | None
    param_count: int
    fp16: bool = False
    dolphin_light: bool = False

def load_dolphin(
    model_id: str = "JusperLee/Dolphin",
    *,
    device: str | None = None,
    warmup: bool = True,
    fp16: bool = False,
    dolphin_light: bool = False,
) -> tuple[Any, str, DolphinLoadStats]:

    ensure_dolphin_import_path()
    import torch

    if dolphin_light:
        from look2hear.models.dolphin_light import Dolphin
    else:
        from look2hear.models.dolphin import Dolphin

    if device is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "requested --device cuda but torch.cuda.is_available() is False"
        )
    if fp16 and not str(dev).startswith("cuda"):
        raise RuntimeError("--fp16 requires CUDA (got device=%r)" % (dev,))

    t0 = time.perf_counter()
    model = Dolphin.from_pretrained(model_id, strict=False)
    model.to(dev)
    if fp16:
        model.half()
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000.0

    dtype = torch.float16 if fp16 else torch.float32
    warmup_ms: float | None = None
    if warmup:
        tw = time.perf_counter()
        mix = torch.zeros(1, SAMPLE_RATE, device=dev, dtype=dtype)
        mouth = torch.zeros(1, 1, MOUTH_FPS, 88, 88, device=dev, dtype=dtype)
        with torch.no_grad():
            _ = model(mix, mouth)
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - tw) * 1000.0

    param_count = sum(p.numel() for p in model.parameters())
    stats = DolphinLoadStats(
        model_id=model_id,
        device=dev,
        load_ms=load_ms,
        warmup_ms=warmup_ms,
        param_count=param_count,
        fp16=fp16,
        dolphin_light=dolphin_light,
    )
    logger.info(
        "loaded %s on %s (%.0f ms, %d params, fp16=%s, light=%s)",
        model_id,
        dev,
        load_ms,
        param_count,
        fp16,
        dolphin_light,
    )
    return model, dev, stats

_NAN_DIAGNOSED = False

def _diagnose_first_nan_module(model: Any, mix_t: Any, mouth_t: Any) -> None:

    import torch

    global _NAN_DIAGNOSED
    if _NAN_DIAGNOSED:
        return
    _NAN_DIAGNOSED = True

    hits: list[str] = []

    def _make_hook(name: str):
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensors = output if isinstance(output, (tuple, list)) else (output,)
            for t in tensors:
                if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all():
                    hits.append(name)
                    break

        return _hook

    handles = [
        module.register_forward_hook(_make_hook(name))
        for name, module in model.named_modules()
        if name
    ]
    try:
        with torch.no_grad():
            model(mix_t, mouth_t)
    finally:
        for h in handles:
            h.remove()

    if hits:
        shown = hits[:8]
        extra = f" (+{len(hits) - 8} more)" if len(hits) > 8 else ""
        print(f"NAN_DIAGNOSTIC: first non-finite submodule outputs, in execution order: {shown}{extra}")
    else:
        print(
            "NAN_DIAGNOSTIC: no registered submodule produced a non-finite output — "
            "the bug is in top-level forward() logic between submodule calls, not inside a submodule."
        )

def infer_separation(
    model: Any,
    mixture: Any,
    mouth: Any,
    *,
    device: str,
    fp16: bool = False,
) -> Any:

    import numpy as np
    import torch

    mix_t = mixture if torch.is_tensor(mixture) else torch.as_tensor(mixture)
    mouth_t = mouth if torch.is_tensor(mouth) else torch.as_tensor(mouth)
    if mix_t.dim() == 1:
        mix_t = mix_t.unsqueeze(0)

    if mouth_t.dim() == 3:
        mouth_t = mouth_t.unsqueeze(0).unsqueeze(0)
    elif mouth_t.dim() == 4 and mouth_t.shape[1] != 1:
        mouth_t = mouth_t.unsqueeze(1)

    dtype = torch.float16 if fp16 else torch.float32
    mix_t = mix_t.to(device=device, dtype=dtype)
    mouth_t = mouth_t.to(device=device, dtype=dtype)

    if fp16:
        _diagnose_first_nan_module(model, mix_t, mouth_t)

    with torch.no_grad():
        est = model(mix_t, mouth_t)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
    out = est.squeeze().detach().float().cpu().numpy()
    return np.asarray(out, dtype=np.float64)

def load_test_dataset(test_dir: str | Path, *, segment: float | None = None) -> Any:

    ensure_dolphin_import_path()
    from look2hear.datas.avspeech_dataset import AVSpeechDataset

    return AVSpeechDataset(
        json_dir=str(test_dir),
        n_src=1,
        sample_rate=SAMPLE_RATE,
        segment=segment,
        normalize_audio=False,
        is_train=False,
    )

def eval_manifest(
    test_dir: str | Path, *, segment: float | None = None
) -> list[dict[str, str]]:

    import json

    root = Path(test_dir)
    with open(root / "mix.json", encoding="utf-8") as f:
        mix_infos = json.load(f)
    with open(root / "s1.json", encoding="utf-8") as f:
        s1_infos = json.load(f)
    with open(root / "s2.json", encoding="utf-8") as f:
        s2_infos = json.load(f)

    seg_len = None if segment is None else int(segment * SAMPLE_RATE)

    items: list[dict[str, str]] = []
    for i in range(len(mix_infos) - 1, -1, -1):
        if seg_len is not None and mix_infos[i][1] < seg_len:
            continue
        mix_path = Path(mix_infos[i][0])
        clip = mix_path.parent.name
        for target, src_info in (("s1", s1_infos[i]), ("s2", s2_infos[i])):
            items.append(
                {
                    "clip_id": f"{clip}/{target}",
                    "target": target,
                    "mix_path": str(mix_path),
                    "ref_path": str(Path(src_info[0])),
                }
            )
    return items
