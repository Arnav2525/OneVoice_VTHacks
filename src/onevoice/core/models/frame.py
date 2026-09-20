from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Frame:

    timestamp_ms: float
    data: Any
    metadata: dict[str, Any]
