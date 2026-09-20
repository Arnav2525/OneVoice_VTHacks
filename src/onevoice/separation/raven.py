

from __future__ import annotations

from typing import Any

from onevoice.separation.adapters.raven_adapter import RavenAdapter
from onevoice.separation.base import LoadableSeparator
from onevoice.separation.config import SeparationConfig
from onevoice.separation.exceptions import BackendInitializationError
from onevoice.separation.utilities import import_torch, resolve_dtype

class RavenSeparator(LoadableSeparator):
    def __init__(self, config: SeparationConfig) -> None:
        super().__init__(config, RavenAdapter())

    def _build_model(self, config: SeparationConfig, device: str) -> Any:
        torch = import_torch()
        if not config.checkpoint:
            raise BackendInitializationError(
                "raven backend requires 'checkpoint' pointing to a scripted model"
            )
        model = torch.jit.load(config.checkpoint, map_location=device)
        model.eval()
        if config.precision != "float32" and device.startswith("cuda"):
            model = model.to(resolve_dtype(config.precision, torch))
        return model
