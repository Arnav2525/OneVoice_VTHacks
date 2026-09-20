

from __future__ import annotations

from typing import Any

from onevoice.separation.adapters.asteroid_adapter import AsteroidAdapter
from onevoice.separation.base import LoadableSeparator
from onevoice.separation.config import SeparationConfig
from onevoice.separation.exceptions import BackendInitializationError
from onevoice.separation.utilities import import_torch, resolve_dtype

class AsteroidSeparator(LoadableSeparator):
    def __init__(self, config: SeparationConfig) -> None:
        super().__init__(config, AsteroidAdapter())

    def _build_model(self, config: SeparationConfig, device: str) -> Any:
        if not config.model_id:
            raise BackendInitializationError(
                "asteroid backend requires 'model_id' (hub id or local path)"
            )
        torch = import_torch()
        try:
            from asteroid.models import BaseModel
        except ImportError as exc:
            raise BackendInitializationError(
                "asteroid is required: pip install asteroid"
            ) from exc
        model = BaseModel.from_pretrained(config.model_id)
        model = model.to(device)
        model.eval()
        if config.precision != "float32" and device.startswith("cuda"):
            model = model.to(resolve_dtype(config.precision, torch))
        return model
