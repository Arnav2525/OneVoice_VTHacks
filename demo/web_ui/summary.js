"use strict";

(() => {
  const el = (name) => document.getElementById(`summary-${name}`);
  const scripted = {
    summary:
      "Two people agreed to meet at noon at the cafe by the library. One will bring the printed schedule and the other will book the table.",
    key_points: [
      "Meet at noon at the cafe by the library",
      "One person brings the printed schedule",
      "The other person books the table",
    ],
  };
  let revision = 0;
  let pending = null;
  let spokenText = "";
  let speaking = null;
  let recordingSummaryKey = null;

  window.onevoiceRecordingSummary = (item) => {
    if (!item) return;
    const key = `${item.path}|${item.status}`;
    if (key === recordingSummaryKey) return;
    recordingSummaryKey = key;
    el("view").textContent = item.status === "loading"
      ? "Summary preparing…"
      : item.status === "ready" ? "View summary" : "Summary needs attention";
    clearResult();
    if (item.status === "ready") {
      show(item.result, false);
      el("status").textContent = "Summary saved with your recording.";
    } else {
      el("status").textContent = item.error || "Gemini is summarizing your recording…";
    }
  };

  function stopSpeech() {
    if (!speaking) return;
    speaking.controller.abort();
    speaking.audio?.pause();
    if (speaking.url) URL.revokeObjectURL(speaking.url);
    speaking = null;
  }

  function controls() {
    el("run").disabled = Boolean(pending);
    el("speak").disabled = !spokenText || Boolean(pending) || Boolean(speaking);
  }

  function clearResult() {
    revision += 1;
    if (pending) pending.abort();
    pending = null;
    stopSpeech();
    spokenText = "";
    el("text").textContent = "";
    el("points").replaceChildren();
    el("badge").hidden = true;
    controls();
  }

  function show(result, isScripted) {
    el("text").textContent = result.summary;
    el("points").replaceChildren();
    result.key_points.forEach((point) => {
      const item = document.createElement("li");
      item.textContent = point;
      el("points").append(item);
    });
    el("badge").textContent = isScripted ? "Scripted demo" : "Gemini summary";
    el("badge").setAttribute("data-status", isScripted ? "scripted" : "gemini");
    el("badge").hidden = false;
    el("status").textContent = isScripted
      ? "Scripted demo · no Gemini request and no captions used"
      : "Generated from the selected person’s captions · check it against what was said";
    spokenText = result.summary;
    controls();
    if (el("auto").checked) speak();
  }

  async function post(path, body, controller) {
    const response = await fetch(path, {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-OneVoice-UI": "1"},
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Request failed.");
    return result;
  }

  async function run() {
    if (pending) return;
    clearResult();
    const epoch = revision;
    const controller = new AbortController();
    pending = controller;
    const timeout = setTimeout(() => controller.abort(), 30000);
    el("status").textContent = "Sending the transcript to Gemini…";
    controls();
    try {
      const result = await post("/api/summarize", {}, controller);
      if (revision !== epoch) return;
      show(result, false);
    } catch (error) {
      if (revision === epoch) {
        el("status").textContent =
          error.name === "AbortError"
            ? "Request timed out. Please retry."
            : error.message;
      }
    } finally {
      clearTimeout(timeout);
      if (revision === epoch) {
        pending = null;
        controls();
      }
    }
  }

  async function speak() {
    if (!spokenText || speaking) return;
    const epoch = revision;
    const controller = new AbortController();
    const current = {controller, audio: null, url: null};
    speaking = current;
    el("status").textContent = "Preparing speech…";
    controls();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const result = await post("/api/speak", {text: spokenText}, controller);
      if (revision !== epoch || speaking !== current) return;
      const bytes = Uint8Array.from(atob(result.audio), (c) => c.charCodeAt(0));
      current.url = URL.createObjectURL(new Blob([bytes], {type: "audio/wav"}));
      current.audio = new Audio(current.url);
      const finished = () => {
        if (speaking !== current) return;
        stopSpeech();
        el("status").textContent = "Finished speaking.";
        controls();
      };
      current.audio.addEventListener("ended", finished);
      current.audio.addEventListener("error", finished);
      el("status").textContent = "Speaking with ElevenLabs…";
      await current.audio.play();
    } catch (error) {
      if (revision === epoch && speaking === current) {
        stopSpeech();
        el("status").textContent =
          error.name === "AbortError"
            ? "Speech timed out. Please retry."
            : error.message;
        controls();
      }
    } finally {
      clearTimeout(timeout);
    }
  }

  el("view").addEventListener("click", () => {
    const opening = el("panel").hidden;
    el("panel").hidden = !opening;
    el("view").setAttribute("aria-expanded", String(opening));
    if (opening) {
      el("panel").scrollIntoView?.({behavior: "smooth", block: "nearest"});
      el("heading").focus?.({preventScroll: true});
    }
  });
  el("run").addEventListener("click", run);
  el("speak").addEventListener("click", speak);
  el("sample").addEventListener("click", () => {
    clearResult();
    show(scripted, true);
  });
  el("clear").addEventListener("click", () => {
    clearResult();
    el("status").textContent = "Summary cleared.";
  });
  window.addEventListener("pagehide", clearResult);
})();
