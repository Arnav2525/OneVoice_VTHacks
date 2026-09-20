

from __future__ import annotations

import importlib
from typing import Any

from onevoice.separation.config import SeparationConfig
from onevoice.separation.exceptions import BackendNotFoundError

_REGISTRY: dict[str, str] = {
    "passthrough": "onevoice.separation.passthrough:PassthroughSeparator",
    "raven": "onevoice.separation.raven:RavenSeparator",
    "clearvoice": "onevoice.separation.clearvoice:ClearVoiceSeparator",
    "speechbrain": "onevoice.separation.speechbrain:SpeechBrainSeparator",
    "asteroid": "onevoice.separation.asteroid:AsteroidSeparator",
    "avtse": "onevoice.separation.avtse:AvtseSeparator",
    "look_once_to_hear": "onevoice.separation.look_once_to_hear:LookOnceToHearSeparator",
    "dolphin": "onevoice.separation.dolphin:DolphinSeparator",
}

def register_backend(name: str, spec: str) -> None:

    _REGISTRY[name.strip().lower()] = spec

def available_backends() -> list[str]:
    return sorted(_REGISTRY)

def resolve_backend_class(name: str) -> type:
    key = name.strip().lower()
    spec = _REGISTRY.get(key)
    if spec is None:
        raise BackendNotFoundError(
            f"unknown backend '{name}'. Available: {available_backends()}"
        )
    module_name, _, class_name = spec.partition(":")
    module = importlib.import_module(module_name)
    backend_cls: type = getattr(module, class_name)
    return backend_cls

def create_separator(raw_config: Any) -> Any:

    config = SeparationConfig.from_raw(raw_config)
    backend_cls = resolve_backend_class(config.name)
    return backend_cls(config)
