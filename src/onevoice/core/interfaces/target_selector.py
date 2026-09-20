from typing import Protocol

from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection

class TargetSelector(Protocol):

    def select_target(
        self, frame: Frame, tracks: list[SpeakerTrack]
    ) -> TargetSelection:

        ...
