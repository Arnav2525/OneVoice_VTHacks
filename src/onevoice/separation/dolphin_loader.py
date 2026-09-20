

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from onevoice.separation.config import SeparationConfig
from onevoice.separation.exceptions import BackendInitializationError
from onevoice.separation.utilities import import_torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MOUTH_FPS = 25
MOUTH_SIZE = 88
DEFAULT_MODEL_ID = "JusperLee/Dolphin"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_VENDOR = _REPO_ROOT / "vendor" / "dolphin"

def vendor_root() -> Path:

    root = Path(os.environ.get("DOLPHIN_VENDOR_ROOT", _DEFAULT_VENDOR))
    if not (root / "look2hear").is_dir():
        raise BackendInitializationError(
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

def _cudnn_convolutions_are_broken(torch_module: Any, device: str) -> bool:

    if not str(device).startswith("cuda"):
        return False
    try:
        probe = torch_module.nn.Conv1d(2, 2, kernel_size=3, padding=1).to(device)

        probe(torch_module.zeros(1, 2, 8, device=device))
        return False
    except RuntimeError as exc:
        if "SUBLIBRARY_VERSION_MISMATCH" not in str(exc):
            raise
        return True

def load_dolphin_model(config: SeparationConfig, device: str) -> Any:

    ensure_dolphin_import_path()
    torch = import_torch()

    if _cudnn_convolutions_are_broken(torch, device):
        logger.warning(
            "cuDNN convolutions are broken on this machine "
            "(CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH probe failed) -- "
            "disabling torch.backends.cudnn for this process. Dolphin will "
            "still run correctly on CUDA, just slower than a machine with a "
            "healthy cuDNN install."
        )
        torch.backends.cudnn.enabled = False

    dolphin_light = bool(config.params.get("dolphin_light", False))
    if dolphin_light:
        from look2hear.models.dolphin_light import Dolphin
    else:
        from look2hear.models.dolphin import Dolphin

    model_id = config.model_id or DEFAULT_MODEL_ID
    model = Dolphin.from_pretrained(model_id, strict=False)
    model.to(device)
    if config.precision == "float16":
        if not str(device).startswith("cuda"):
            raise BackendInitializationError(
                "dolphin precision=float16 requires a CUDA device"
            )
        model.half()
    model.eval()
    return model

def resolve_dtype(config: SeparationConfig, torch_module: Any) -> Any:
    return torch_module.float16 if config.precision == "float16" else torch_module.float32
