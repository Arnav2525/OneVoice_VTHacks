

from __future__ import annotations

from typing import Any

from onevoice.separation.adapters.speechbrain_adapter import SpeechBrainAdapter
from onevoice.separation.base import LoadableSeparator
from onevoice.separation.config import SeparationConfig
from onevoice.separation.exceptions import BackendInitializationError

DEFAULT_MODEL_ID = "speechbrain/sepformer-wsj02mix"

class SpeechBrainSeparator(LoadableSeparator):
    def __init__(self, config: SeparationConfig) -> None:
        super().__init__(config, SpeechBrainAdapter())

    def _build_model(self, config: SeparationConfig, device: str) -> Any:
        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError as exc:
            raise BackendInitializationError(
                "speechbrain is required for the speechbrain backend: "
                "pip install -e '.[speechbrain]'"
            ) from exc

        source = config.checkpoint or config.model_id or DEFAULT_MODEL_ID
        savedir = config.params.get("savedir", f"pretrained_models/{config.name}")
        return SepformerSeparation.from_hparams(
            source=source,
            savedir=savedir,
            run_opts={"device": device},
        )
