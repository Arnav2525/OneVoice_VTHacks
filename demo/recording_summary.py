from __future__ import annotations

import json
import threading

from demo.summary import MAX_TRANSCRIPT_CHARS, summarize


class RecordingSummaries:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}
        self.latest = None

    def snapshot(self):
        with self.lock:
            return dict(self.jobs[self.latest]) if self.latest else None

    def start(self, path, transcript):
        key = str(path)
        with self.lock:
            if key in self.jobs:
                return
            self.latest = key
            self.jobs[key] = {
                "path": key,
                "status": "loading",
                "result": None,
                "error": None,
            }
        worker = threading.Thread(
            target=self._run,
            args=(path, transcript),
            daemon=False,
            name="recording-summary",
        )
        worker.start()
        return worker

    def _run(self, path, transcript):
        state = {"path": str(path), "status": "ready", "result": None, "error": None}
        try:
            if not transcript.strip():
                raise ValueError(
                    "No finalized captions were captured. "
                    "Enable Captions before recording."
                )
            if len(transcript) <= MAX_TRANSCRIPT_CHARS:
                result = summarize(transcript)
            else:
                sections = []
                for offset in range(0, len(transcript), MAX_TRANSCRIPT_CHARS):
                    part = summarize(transcript[offset : offset + MAX_TRANSCRIPT_CHARS])
                    sections.append(
                        part["summary"] + "\n" + "\n".join(part["key_points"])
                    )
                combined = "\n\n".join(sections)
                if len(combined) > MAX_TRANSCRIPT_CHARS:
                    raise ValueError(
                        "Conversation is too long for one summary. "
                        "Saved transcript is complete."
                    )
                result = summarize(combined)
            text = result["summary"] + "\n"
            if result["key_points"]:
                text += (
                    "\nKey points\n"
                    + "\n".join("- " + point for point in result["key_points"])
                    + "\n"
                )
            (path / "summary.txt").write_text(text, encoding="utf-8")
            state["result"] = result
        except (OSError, RuntimeError, ValueError) as exc:
            state.update(status="error", error=str(exc))
        try:
            (path / "summary_status.json").write_text(
                json.dumps(state, indent=2), encoding="utf-8"
            )
        except OSError:
            state.update(status="error", error="Could not save the summary to disk.")
        with self.lock:
            self.jobs[str(path)] = state
