from demo.recording_summary import RecordingSummaries


def test_stop_recording_triggers_summary_after_pending_captions(tmp_path):
    from unittest.mock import Mock

    from demo.session_runtime import SessionRunner

    runner = SessionRunner({}, record_root=tmp_path)
    runner.recording_summaries = Mock()
    runner.captions.finish_pending = lambda: runner.recorder.on_transcript(
        "Person 1", "Last sentence."
    )
    path = runner.recorder.start_clip(tmp_path, "live")
    runner._change_recording(True)
    runner.recording_summaries.start.assert_called_once_with(
        path, "Person 1: Last sentence.\n"
    )
    runner._finish_recording()
    assert runner.recording_summaries.start.call_count == 1


def test_summary_saved_once_and_input_is_frozen(tmp_path, monkeypatch):
    calls = []

    def summarize(text):
        calls.append(text)
        return {"summary": "They planned a test.", "key_points": ["Tomorrow"]}

    monkeypatch.setattr("demo.recording_summary.summarize", summarize)
    summaries = RecordingSummaries()
    worker = summaries.start(tmp_path, "Person 1: Test tomorrow.")
    worker.join(2)
    assert summaries.start(tmp_path, "Different transcript") is None
    assert calls == ["Person 1: Test tomorrow."]
    assert "Tomorrow" in (tmp_path / "summary.txt").read_text()
    assert summaries.snapshot()["status"] == "ready"


def test_failed_summary_preserves_transcript(tmp_path, monkeypatch):
    (tmp_path / "transcript.txt").write_text("Hello")

    def fail(text):
        raise RuntimeError("Set GEMINI_API_KEY on the server, then retry.")

    monkeypatch.setattr("demo.recording_summary.summarize", fail)
    summaries = RecordingSummaries()
    summaries.start(tmp_path, "Hello").join(2)
    assert summaries.snapshot()["status"] == "error"
    assert "GEMINI_API_KEY" in summaries.snapshot()["error"]
    assert (tmp_path / "transcript.txt").read_text() == "Hello"
    assert not (tmp_path / "summary.txt").exists()


def test_empty_transcript_does_not_call_api(tmp_path, monkeypatch):
    def unexpected(text):
        raise AssertionError("No API call expected")

    monkeypatch.setattr("demo.recording_summary.summarize", unexpected)
    summaries = RecordingSummaries()
    summaries.start(tmp_path, "").join(2)
    assert summaries.snapshot()["status"] == "error"


def test_long_conversation_is_not_silently_truncated(tmp_path, monkeypatch):
    calls = []

    def summarize(text):
        calls.append(text)
        return {"summary": "Brief section.", "key_points": []}

    monkeypatch.setattr("demo.recording_summary.summarize", summarize)
    summaries = RecordingSummaries()
    text = "x" * 12000 + "Last sentence."
    summaries.start(tmp_path, text).join(2)
    assert calls[0] + calls[1] == text
    assert len(calls) == 3
    assert summaries.snapshot()["status"] == "ready"
