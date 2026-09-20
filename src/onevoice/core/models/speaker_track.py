from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class SpeakerTrack:

    track_id: str
    bounding_box: tuple[float, float, float, float]
    confidence: float
    metadata: dict[str, Any]
