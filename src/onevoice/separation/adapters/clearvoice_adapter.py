

from __future__ import annotations

from typing import Any

from onevoice.separation.adapters.base import AdapterContext, SeparatorAdapter
from onevoice.separation.utilities import import_torch

class ClearVoiceAdapter(SeparatorAdapter):
    def infer(self, model: Any, backend_input: Any, ctx: AdapterContext) -> Any:
        torch = import_torch()
        with torch.no_grad():
            output = model(backend_input)
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output
