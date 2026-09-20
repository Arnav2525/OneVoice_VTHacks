

from __future__ import annotations

from typing import Any

from onevoice.separation.adapters.clearvoice_adapter import ClearVoiceAdapter
from onevoice.separation.base import LoadableSeparator
from onevoice.separation.config import SeparationConfig
from onevoice.separation.exceptions import BackendInitializationError

class ClearVoiceSeparator(LoadableSeparator):
    def __init__(self, config: SeparationConfig) -> None:
        super().__init__(config, ClearVoiceAdapter())

    def _build_model(self, config: SeparationConfig, device: str) -> Any:
        try:
            from clearvoice import ClearVoice
        except ImportError as exc:
            raise BackendInitializationError(
                "clearvoice is required: pip install clearvoice"
            ) from exc
        task = config.params.get("task", "target_speaker_extraction")
        model_names = config.params.get("model_names")
        if not model_names:
            raise BackendInitializationError(
                "clearvoice backend requires params.model_names "
                "(e.g. ['MossFormer2_SE_48K'])"
            )
        cv = ClearVoice(task=task, model_names=model_names)
        network = getattr(cv, "network", None) or getattr(cv, "model", None)
        if network is None:
            raise BackendInitializationError(
                "could not resolve a tensor-callable network from ClearVoice; "
                "set params.callable to expose one"
            )
        network = network.to(device)
        network.eval()
        return network
