

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from onevoice.separation.config import SeparationConfig
from onevoice.separation.exceptions import BackendInitializationError
from onevoice.separation.utilities import import_torch

logger = logging.getLogger(__name__)

DEFAULT_LOTH_ROOT = Path.home() / ".cache" / "onevoice" / "LookOnceToHear"
DEFAULT_TSH_CONFIG = "configs/tsh.json"
DEFAULT_TSH_CHECKPOINT = "runs/tsh/best.ckpt"
DEFAULT_EMBED_CONFIG = "configs/embed.json"
DEFAULT_EMBED_CHECKPOINT = "runs/embed/best.ckpt"

@dataclass
class LothModels:

    net: Any
    embed_net: Any

def resolve_loth_root(config: SeparationConfig) -> Path:
    raw = config.params.get("loth_root") or config.params.get("repo_path")
    if raw:
        return Path(str(raw)).expanduser().resolve()
    env = __import__("os").environ.get("ONEVOICE_LOTH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_LOTH_ROOT

def resolve_checkpoint(
    config: SeparationConfig,
    root: Path,
    *,
    rel_key: str,
    default_rel: str,
) -> Path:
    if rel_key == "checkpoint_rel" and config.checkpoint:
        path = Path(config.checkpoint).expanduser()
        if path.is_file():
            return path.resolve()
    rel = config.params.get(rel_key, default_rel)
    candidate = root / str(rel)
    if candidate.is_file():
        return candidate.resolve()
    raise BackendInitializationError(
        f"LookOnceToHear checkpoint not found at {candidate}. "
        "See scripts/setup_loth.py for download instructions."
    )

def resolve_config_path(
    config: SeparationConfig,
    root: Path,
    *,
    path_key: str,
    rel_key: str,
    default_rel: str,
) -> Path:
    if config.params.get(path_key):
        path = Path(str(config.params[path_key])).expanduser()
        if path.is_file():
            return path.resolve()
    rel = config.params.get(rel_key, default_rel)
    candidate = root / rel
    if candidate.is_file():
        return candidate.resolve()
    raise BackendInitializationError(
        f"LookOnceToHear config not found at {candidate}. "
        "Run: python scripts/setup_loth.py"
    )

def _ensure_on_path(root: Path) -> None:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

def _import_attr(dotted: str) -> type:
    module_name, _, attr = dotted.rpartition(".")
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)

def _load_pl_module(cfg_path: Path, ckpt_path: Path, device: str) -> Any:
    torch = import_torch()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    pl_cls = _import_attr(cfg["pl_module"])
    pl_module = pl_cls(**cfg["pl_module_args"])
    logger.info("loading LookOnceToHear checkpoint %s", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    pl_module.load_state_dict(state, strict=False)
    pl_module.eval()
    pl_module.to(device)
    return pl_module.model

def load_look_once_model(config: SeparationConfig, device: str) -> LothModels:

    root = resolve_loth_root(config)
    if not root.is_dir():
        raise BackendInitializationError(
            f"LookOnceToHear source tree not found at {root}. "
            "Run: python scripts/setup_loth.py"
        )
    _ensure_on_path(root)
    tsh_cfg = resolve_config_path(
        config,
        root,
        path_key="config_path",
        rel_key="config_rel",
        default_rel=DEFAULT_TSH_CONFIG,
    )
    tsh_ckpt = resolve_checkpoint(
        config,
        root,
        rel_key="checkpoint_rel",
        default_rel=DEFAULT_TSH_CHECKPOINT,
    )
    embed_cfg = resolve_config_path(
        config,
        root,
        path_key="embed_config_path",
        rel_key="embed_config_rel",
        default_rel=DEFAULT_EMBED_CONFIG,
    )
    embed_ckpt = resolve_checkpoint(
        config,
        root,
        rel_key="embed_checkpoint_rel",
        default_rel=DEFAULT_EMBED_CHECKPOINT,
    )
    net = _load_pl_module(tsh_cfg, tsh_ckpt, device)
    embed_net = _load_pl_module(embed_cfg, embed_ckpt, device)
    net.eval()
    embed_net.eval()
    return LothModels(net=net, embed_net=embed_net)
