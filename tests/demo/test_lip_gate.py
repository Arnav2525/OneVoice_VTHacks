import numpy as np

from demo.lip_gate import LipGate
from onevoice.core.models.audio_chunk import AudioChunk


def audio(source, count=640, **metadata):
    return AudioChunk(9000, [1.0] * count, 16000, 1, {
        "source_timestamp_ms": source, "target_track_id": "p1",
        "ui_selection_epoch": 1, **metadata,
    })


def observed(gate, resume=None):
    for t in range(0, 2001, 40):
        aperture = 0.1 if t < 200 or (resume is not None and t >= resume) else 0.0
        gate.observe(t, ("p1", 1), aperture)


def test_short_pause_is_preserved_and_sustained_pause_is_quiet():
    gate = LipGate()
    observed(gate)
    assert min(gate.process(audio(400)).data) == 1.0
    assert max(gate.process(audio(1000)).data[-320:]) < 0.02


def test_resume_restores_speech_without_resetting_selection():
    gate = LipGate()
    observed(gate, resume=1200)
    gate.process(audio(900))
    assert min(gate.process(audio(1160)).data[-320:]) > 0.99


def test_uses_audio_content_time_not_playback_time():
    gate = LipGate()
    observed(gate)
    assert min(gate.process(audio(700, 160, denoise_delay_ms=32)).data) == 1.0


def test_missing_stale_or_new_identity_does_not_mute():
    gate = LipGate()
    observed(gate)
    assert min(gate.process(audio(None)).data) == 1.0
    assert min(gate.process(audio(5000)).data) == 1.0
    assert min(gate.process(audio(1000, ui_selection_epoch=2)).data) == 1.0
    gate.observe(2040, ("p1", 1), None)
    assert min(gate.process(audio(2040)).data) == 1.0


def test_processing_is_independent_of_chunk_boundaries():
    whole, split = LipGate(), LipGate()
    observed(whole, 1200)
    observed(split, 1200)
    expected = whole.process(audio(600, 16000)).data
    actual = []
    for offset in range(0, 16000, 320):
        actual.extend(split.process(audio(600 + offset / 16, 320)).data)
    np.testing.assert_allclose(actual, expected)


def test_alarm_bypasses_closed_lip_gate():
    from unittest.mock import Mock

    from tests.demo.test_session_safety import FakeMonitor, prepared_sink

    sink, _, chunk, destination, _, _ = prepared_sink(
        safety=FakeMonitor(active=True, raw=audio(1000))
    )
    gate = Mock()
    gate.process.side_effect = AssertionError("Alarm must bypass lip suppression")
    sink.lip_gate = gate
    sink.write(chunk)
    assert any(destination[-1].data)
    gate.process.assert_not_called()
    gate.reset.assert_called_once()
