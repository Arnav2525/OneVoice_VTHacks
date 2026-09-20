from typing import Protocol

from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack

class FaceTracker(Protocol):

    def process_frame(self, frame: Frame) -> list[SpeakerTrack]:

        ...
