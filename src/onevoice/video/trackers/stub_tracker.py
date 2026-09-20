

from __future__ import annotations

from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack

class StubFaceTracker:

    def __init__(self, track_id: str = "stub-0", confidence: float = 0.9) -> None:
        self._track_id = track_id
        self._confidence = confidence

    def process_frame(self, frame: Frame) -> list[SpeakerTrack]:
        width = float(frame.metadata.get("width", 640))
        height = float(frame.metadata.get("height", 480))
        box_w = width * 0.25
        box_h = height * 0.35
        x = (width - box_w) / 2.0
        y = (height - box_h) / 2.0
        return [
            SpeakerTrack(
                track_id=self._track_id,
                bounding_box=(x, y, box_w, box_h),
                confidence=self._confidence,
                metadata={"stub": True},
            )
        ]
