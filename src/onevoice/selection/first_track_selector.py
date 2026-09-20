

from __future__ import annotations

from onevoice.core.models.frame import Frame
from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.core.models.target_selection import TargetSelection

class FirstTrackSelector:

    def select_target(
        self, frame: Frame, tracks: list[SpeakerTrack]
    ) -> TargetSelection:
        if not tracks:
            return TargetSelection(
                timestamp_ms=frame.timestamp_ms, selected_speaker=None
            )
        best = max(tracks, key=lambda t: t.confidence)
        return TargetSelection(timestamp_ms=frame.timestamp_ms, selected_speaker=best)
