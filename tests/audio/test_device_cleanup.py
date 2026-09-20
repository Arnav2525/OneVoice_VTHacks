

from types import SimpleNamespace

import pytest

from onevoice.audio import io
from onevoice.video import capture

class _Stream:
    def __init__(self, *, fail_start=False, fail_stop=False, fail_close=False):
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.fail_close = fail_close
        self.stop_calls = 0
        self.close_calls = 0
        self.stop_error = RuntimeError("Stop failed")

    def start(self):
        if self.fail_start:
            raise RuntimeError("Start failed after allocating handle")

    def stop(self):
        self.stop_calls += 1
        if self.fail_stop:
            raise self.stop_error

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("Close failed")

def _audio_device(monkeypatch, device_type, stream):
    monkeypatch.setattr(
        io,
        "sd",
        SimpleNamespace(
            InputStream=lambda **kwargs: stream,
            OutputStream=lambda **kwargs: stream,
        ),
    )
    return device_type()

@pytest.mark.parametrize("device_type", [io.MicrophoneSource, io.SpeakerSink])
def test_partially_started_audio_stream_is_closed(monkeypatch, device_type):
    stream = _Stream(fail_start=True)
    device = _audio_device(monkeypatch, device_type, stream)
    with pytest.raises(RuntimeError, match="Start failed"):
        device.start()
    device.stop()
    device.stop()
    assert stream.close_calls == 1
    assert device._stream is None

@pytest.mark.parametrize("device_type", [io.MicrophoneSource, io.SpeakerSink])
@pytest.mark.parametrize("fail_close", [False, True])
def test_audio_close_runs_even_if_stop_fails(monkeypatch, device_type, fail_close):
    stream = _Stream(fail_stop=True, fail_close=fail_close)
    device = _audio_device(monkeypatch, device_type, stream)
    device.start()
    with pytest.raises(RuntimeError) as error:
        device.stop()
    assert error.value is stream.stop_error
    assert stream.close_calls == 1
    assert not device._running
    assert (device._stream is stream) is fail_close
    stream.fail_stop = stream.fail_close = False
    device.stop()
    assert device._stream is None

def test_failed_webcam_open_releases_capture_handle(monkeypatch):
    released = []
    webcam = SimpleNamespace(
        isOpened=lambda: False, release=lambda: released.append(True)
    )
    monkeypatch.setattr(capture, "cv2", SimpleNamespace(VideoCapture=lambda _: webcam))
    device = capture.WebcamSource()
    with pytest.raises(RuntimeError, match="Failed to open webcam"):
        device.start()
    device.stop()
    device.stop()
    assert released == [True]
    assert device._capture is None

@pytest.mark.parametrize("wrapper_type", [io.TeeAudioSink, io.RecordingAudioSource])
def test_wrapped_devices_all_stop_even_if_first_fails(wrapper_type):
    calls = []
    first_error = RuntimeError("First device cleanup failed")

    def fail():
        calls.append("first")
        raise first_error

    first = SimpleNamespace(stop=fail)
    second = SimpleNamespace(stop=lambda: calls.append("second"))
    if wrapper_type is io.TeeAudioSink:
        wrapper = wrapper_type([first, second])
    else:
        wrapper = wrapper_type(first, second)
    with pytest.raises(RuntimeError) as error:
        wrapper.stop()
    assert error.value is first_error
    assert calls == ["first", "second"]
