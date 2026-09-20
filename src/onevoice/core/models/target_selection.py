from dataclasses import dataclass

from .speaker_track import SpeakerTrack

@dataclass(frozen=True)
class TargetSelection:

    timestamp_ms: float
    selected_speaker: SpeakerTrack | None
