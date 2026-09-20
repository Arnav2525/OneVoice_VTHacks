

from __future__ import annotations

from onevoice.core.models.speaker_track import SpeakerTrack
from onevoice.video.asd.motion_scorer import AsdConfig, MotionActiveSpeakerScorer

def _track(tid: str, mouth: tuple[float, float]) -> SpeakerTrack:
    return SpeakerTrack(
        track_id=tid,
        bounding_box=(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        metadata={"landmarks": {"mouth": mouth}},
    )

def test_motion_increases_with_movement():
    scorer = MotionActiveSpeakerScorer(AsdConfig(history_len=4, motion_scale=5.0))
    still = scorer.score(_track("a", (50.0, 50.0)))
    moving = scorer.score(_track("a", (55.0, 50.0)))
    assert moving > still

def test_neutral_without_landmarks():
    scorer = MotionActiveSpeakerScorer()
    t = SpeakerTrack("x", (0, 0, 1, 1), 0.5, metadata={})
    assert scorer.score(t) == 0.0
