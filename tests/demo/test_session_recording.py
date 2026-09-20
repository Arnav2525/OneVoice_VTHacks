

from __future__ import annotations

import array
import json
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo.session_recording import SessionRecorder  # noqa: E402

def chunk(samples=(0.25, -0.5), timestamp=1000.0, rate=16000, channels=1, **metadata):
    return SimpleNamespace(
        data=array.array("f", samples),
        timestamp_ms=timestamp,
        sample_rate=rate,
        channels=channels,
        metadata=metadata,
    )

def manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="utf-8"))

def test_inactive_taps_do_not_create_any_files(tmp_path):
    recorder = SessionRecorder()
    recorder.on_input(chunk())
    recorder.on_output(chunk())
    assert not list(tmp_path.iterdir())
    assert recorder.stop_clip() is None
    assert recorder.snapshot().path is None

def test_real_wavs_and_manifest_preserve_distinct_stream_timing(tmp_path):
    recorder = SessionRecorder()
    path = recorder.start_clip(tmp_path, "preview")
    recorder.on_input(chunk(timestamp=1500.0))
    recorder.on_input(chunk(timestamp=1700.0))
    recorder.on_output(chunk(timestamp=1520.0, passthrough=True))
    assert recorder.stop_clip() == path
    assert not recorder.snapshot().active
    assert recorder.snapshot().error is None
    info = manifest(path)
    assert info["status"] == "completed"
    assert "not evidence of speech isolation" in info["output_description"]
    assert not (path / "isolated.wav").exists()
    for stream, frames in (("input", 4), ("output", 2)):
        with wave.open(str(path / f"{stream}_audio.wav"), "rb") as wav:
            assert wav.getparams()[:4] == (1, 2, 16000, frames)
            assert len(wav.readframes(frames)) == frames * 2
    raw = info["streams"]["input"]["chunks"]
    assert [part["source_timestamp_ms"] for part in raw] == [1500.0, 1700.0]
    assert [part["wav_frame_offset"] for part in raw] == [0, 2]
    output = info["streams"]["output"]["chunks"][0]
    assert output["source_timestamp_ms"] == 1520.0
    assert output["metadata"]["passthrough"] is True
    assert output["arrival_offset_ms"] >= 0

def test_repeated_record_creates_unique_directories_without_overwriting(tmp_path):
    recorder = SessionRecorder()
    first = recorder.start_clip(tmp_path, "live")
    assert recorder.start_clip(tmp_path, "live") == first
    recorder.on_input(chunk())
    recorder.stop_clip()
    original = (first / "input_audio.wav").read_bytes()
    second = recorder.start_clip(tmp_path, "preview")
    recorder.on_input(chunk(samples=(1.0,) * 10))
    recorder.stop_clip()
    assert first != second
    assert (first / "input_audio.wav").read_bytes() == original
    assert manifest(second)["mode"] == "preview"

def test_manifest_keeps_device_filter_and_selection_evidence(tmp_path):
    recorder = SessionRecorder()
    path = recorder.start_clip(tmp_path, "live")
    recorder.on_input(
        chunk(input_device=6, input_device_requested="C270", overflowed=False)
    )
    recorder.on_output(
        chunk(
            target_track_id="face-2",
            ui_selection_epoch=4,
            output_ready=True,
            ui_muted=False,
            noise_suppression="gtcrn",
            denoise_delay_ms=32.0,
            visual_coverage=0.96,
            buffered_delay_ms=2000.0,
            source_timestamp_ms=1234.0,
            rumble_filter_hz=float("nan"),
        )
    )
    recorder.stop_clip()
    streams = manifest(path)["streams"]
    assert streams["input"]["chunks"][0]["metadata"]["input_device"] == 6
    metadata = streams["output"]["chunks"][0]["metadata"]
    assert metadata["visual_coverage"] == 0.96
    assert metadata["ui_selection_epoch"] == 4
    assert metadata["noise_suppression"] == "gtcrn"
    assert metadata["denoise_delay_ms"] == 32.0
    assert "rumble_filter_hz" not in metadata

def test_stereo_sample_frames_and_clipping_are_correct(tmp_path):
    recorder = SessionRecorder()
    path = recorder.start_clip(tmp_path, "live")
    recorder.on_output(chunk(samples=(-2.0, 2.0, 0.0, 0.5), channels=2))
    recorder.stop_clip()
    with wave.open(str(path / "output_audio.wav"), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getnframes() == 2
        pcm = array.array("h", wav.readframes(2))
        if sys.byteorder != "little":
            pcm.byteswap()
        assert pcm.tolist() == [-32768, 32767, 0, 16383]
    assert manifest(path)["streams"]["output"]["frames_written"] == 2

def test_queue_overflow_stops_visibly_and_drains_accepted_audio(tmp_path, monkeypatch):
    recorder = SessionRecorder(max_queue_chunks=1)
    entered, release = threading.Event(), threading.Event()
    original = recorder._write_packet

    def slow_write(*args):
        entered.set()
        assert release.wait(timeout=3)
        original(*args)

    monkeypatch.setattr(recorder, "_write_packet", slow_write)
    path = recorder.start_clip(tmp_path, "preview")
    try:
        recorder.on_input(chunk())
        assert entered.wait(timeout=3)
        recorder.on_input(chunk())
        recorder.on_input(chunk())
        assert not recorder.snapshot().active
        assert "queue is full" in recorder.snapshot().error
    finally:
        release.set()
        recorder.stop_clip()
    info = manifest(path)
    assert info["status"] == "incomplete"
    assert info["streams"]["input"]["frames_written"] == 4

def test_writer_failure_does_not_escape_audio_taps(tmp_path, monkeypatch):
    recorder = SessionRecorder()

    def failed_write(*args):
        raise OSError("disk full")

    monkeypatch.setattr(recorder, "_write_packet", failed_write)
    path = recorder.start_clip(tmp_path, "live")
    recorder.on_input(chunk())
    recorder.stop_clip()
    recorder.on_output(chunk())
    assert "disk full" in recorder.snapshot().error
    assert manifest(path)["status"] == "incomplete"

def test_setup_failure_is_visible_and_can_recover(tmp_path):
    recorder = SessionRecorder()
    occupied = tmp_path / "not_a_directory"
    occupied.write_text("keep me")
    with pytest.raises(OSError):
        recorder.start_clip(occupied, "preview")
    assert not recorder.snapshot().active
    assert "Could not start recording" in recorder.snapshot().error
    recorder.on_input(chunk())
    path = recorder.start_clip(tmp_path, "preview")
    recorder.stop_clip()
    assert recorder.snapshot().error is None
    assert manifest(path)["streams"] == {}
    assert occupied.read_text() == "keep me"

def test_format_change_and_invalid_audio_stop_with_error(tmp_path):
    recorder = SessionRecorder()
    path = recorder.start_clip(tmp_path, "live")
    recorder.on_input(chunk())
    recorder.on_input(chunk(rate=8000))
    recorder.stop_clip()
    assert "format changed" in recorder.snapshot().error
    with wave.open(str(path / "input_audio.wav"), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 2
    recorder.start_clip(tmp_path, "preview")
    recorder.on_output(chunk(samples=(1.0,), channels=2))
    recorder.stop_clip()
    assert "Invalid audio" in recorder.snapshot().error

def test_capture_buffer_is_owned_before_background_write(tmp_path, monkeypatch):
    recorder = SessionRecorder()
    entered, release = threading.Event(), threading.Event()
    original = recorder._write_packet

    def slow_write(*args):
        entered.set()
        assert release.wait(timeout=3)
        original(*args)

    monkeypatch.setattr(recorder, "_write_packet", slow_write)
    path = recorder.start_clip(tmp_path, "live")
    packet = chunk(samples=(0.5,))
    try:
        recorder.on_input(packet)
        assert entered.wait(timeout=3)
        packet.data[0] = 0.0
    finally:
        release.set()
        recorder.stop_clip()
    with wave.open(str(path / "input_audio.wav"), "rb") as wav:
        assert wav.readframes(1) == b"\xff\x3f"
