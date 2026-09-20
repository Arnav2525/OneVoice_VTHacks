

from __future__ import annotations

import array
import wave

from onevoice.audio.io import RecordingAudioSource, TeeAudioSink, WavFileSink
from onevoice.core.models.audio_chunk import AudioChunk

def _chunk(value: float, samples: int = 16, ts: float = 0.0) -> AudioChunk:
    return AudioChunk(
        timestamp_ms=ts,
        data=array.array("f", [value] * samples),
        sample_rate=16000,
        channels=1,
        metadata={},
    )

def _read_wav_samples(path) -> list[int]:
    with wave.open(str(path), "rb") as wav:
        assert wav.getsampwidth() == 2
        raw = wav.readframes(wav.getnframes())
        return list(array.array("h", raw))

class _FakeSink:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.written = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def write(self, chunk):
        self.written.append(chunk)

class _FakeSource:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read(self):
        return self._chunks.pop(0)

def test_wav_file_sink_writes_correct_samples(tmp_path):
    path = tmp_path / "out.wav"
    sink = WavFileSink(path, sample_rate=16000, channels=1)
    sink.start()
    sink.write(_chunk(0.5, samples=4))
    sink.write(_chunk(-1.0, samples=4))
    sink.stop()

    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getnframes() == 8

    samples = _read_wav_samples(path)
    assert samples[:4] == [16383] * 4
    assert samples[4:] == [-32767] * 4

def test_wav_file_sink_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "out.wav"
    sink = WavFileSink(path, sample_rate=16000, channels=1)
    sink.start()
    sink.write(_chunk(0.0, samples=4))
    sink.stop()
    assert path.is_file()

def test_tee_audio_sink_forwards_to_all_sinks():
    a, b = _FakeSink(), _FakeSink()
    tee = TeeAudioSink([a, b])
    tee.start()
    chunk = _chunk(0.3)
    tee.write(chunk)
    tee.stop()
    assert a.started and b.started
    assert a.stopped and b.stopped
    assert a.written == [chunk]
    assert b.written == [chunk]

def test_tee_audio_sink_with_real_wav_file(tmp_path):
    path = tmp_path / "recorded.wav"
    playback = _FakeSink()
    recorder = WavFileSink(path, sample_rate=16000, channels=1)
    tee = TeeAudioSink([playback, recorder])
    tee.start()
    tee.write(_chunk(0.25, samples=8))
    tee.stop()
    assert len(playback.written) == 1
    assert _read_wav_samples(path) == [8191] * 8

def test_recording_audio_source_records_and_passes_through(tmp_path):
    path = tmp_path / "input.wav"
    chunks = [_chunk(0.1, samples=4, ts=0.0), _chunk(0.2, samples=4, ts=20.0)]
    source = _FakeSource(chunks)
    recorder = WavFileSink(path, sample_rate=16000, channels=1)
    rec_source = RecordingAudioSource(source, recorder)

    rec_source.start()
    first = rec_source.read()
    second = rec_source.read()
    rec_source.stop()

    assert source.started and source.stopped
    assert first.timestamp_ms == 0.0
    assert second.timestamp_ms == 20.0

    with wave.open(str(path), "rb") as wav:
        assert wav.getnframes() == 8
