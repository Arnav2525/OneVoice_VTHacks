from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class AudioChunk:

    timestamp_ms: float
    data: Any
    sample_rate: int
    channels: int
    metadata: dict[str, Any]
