

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from onevoice.core.models.frame import Frame

Landmarks = dict[str, tuple[float, float]]

@dataclass(frozen=True)
class Detection:

    bounding_box: tuple[float, float, float, float]
    confidence: float
    landmarks: Landmarks | None = None
    lip_bbox: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

class FaceDetector(Protocol):

    def detect(self, frame: Frame) -> list[Detection]:

        ...
