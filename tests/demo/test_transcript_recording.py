from demo.elevenlabs_transcription import ElevenLabsCaptions
from demo.session_recording import SessionRecorder
from tests.demo.test_elevenlabs_captions import FakeStream
from tests.demo.test_session_captions import _chunk, _ready_state
from tests.demo.test_session_recording import chunk, manifest


def test_full_session_transcript_saved_beside_audio_and_keeps_late_finals(tmp_path):
    recorder = SessionRecorder()
    state = _ready_state()
    state.select("a")
    captions = ElevenLabsCaptions(
        state, stream_factory=FakeStream, on_final=recorder.on_transcript
    )
    captions.toggle()
    captions.on_output(_chunk(*state.selection()), True)
    stream = captions.stream
    stream.callback("Before recording.", True)
    path = recorder.start_clip(tmp_path, "live")
    recorder.on_input(chunk())
    recorder.on_output(chunk())
    stream.callback("unfinished", False)
    stream.callback("During recording.", True)
    recorder.stop_clip()
    captions.retire_current_sink()
    stream.callback("Final sentence.", True)
    text = (path / "transcript.txt").read_text(encoding="utf-8")
    assert text == (
        "Person 1: Before recording.\n"
        "Person 1: During recording.\n"
        "Person 1: Final sentence.\n"
    )
    assert (path / "input_audio.wav").exists()
    assert (path / "output_audio.wav").exists()
    assert manifest(path)["transcript"]["file"] == "transcript.txt"


def test_multiple_clips_share_session_but_new_session_starts_fresh(tmp_path):
    recorder = SessionRecorder()
    first = recorder.start_clip(tmp_path, "live")
    recorder.on_transcript("Person 1", "Hello.")
    recorder.stop_clip()
    second = recorder.start_clip(tmp_path, "live")
    recorder.on_transcript("Person 2", "Goodbye.")
    recorder.stop_clip()
    expected = "Person 1: Hello.\nPerson 2: Goodbye.\n"
    for path in (first, second):
        assert (path / "transcript.txt").read_text(encoding="utf-8") == expected
    recorder.begin_transcript_session()
    third = recorder.start_clip(tmp_path, "live")
    recorder.on_transcript("Person 1", "New conversation.")
    recorder.stop_clip()
    assert (first / "transcript.txt").read_text(encoding="utf-8") == expected
    assert (third / "transcript.txt").read_text(encoding="utf-8") == (
        "Person 1: New conversation.\n"
    )


def test_transcript_is_not_limited_by_summary_history(tmp_path):
    recorder = SessionRecorder()
    for index in range(350):
        recorder.on_transcript("Person 1", f"Sentence {index}: café.")
    path = recorder.start_clip(tmp_path, "live")
    recorder.stop_clip()
    lines = (path / "transcript.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 350
    assert "Sentence 0: café." in lines[0]
    assert "Sentence 349: café." in lines[-1]


def test_recording_without_captions_has_empty_transcript(tmp_path):
    recorder = SessionRecorder()
    path = recorder.start_clip(tmp_path, "live")
    recorder.stop_clip()
    assert (path / "transcript.txt").read_text(encoding="utf-8") == ""
