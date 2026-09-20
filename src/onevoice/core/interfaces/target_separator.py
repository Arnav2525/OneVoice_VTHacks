from typing import Protocol

from onevoice.core.models.audio_chunk import AudioChunk
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection

class TargetSeparator(Protocol):

    def separate(
        self,
        audio_chunk: AudioChunk,
        target: TargetSelection,
        all_tracks: list[SpeakerTrack]
    ) -> AudioChunk:

        ...
