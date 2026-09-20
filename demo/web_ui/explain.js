"use strict";

(() => {
  const el = (name) => document.getElementById(`explain-${name}`);
  const canvas = el("canvas");
  const context = canvas.getContext("2d");
  const sentence = "Show me the blue cable.";
  let frozen = null;
  let sample = false;
  let revision = 0;
  let pending = null;
  let source = "";
  let spokenText = "";
  let speaking = null;

  function clearResult() {
    revision += 1;
    if (pending) pending.abort();
    pending = null;
    stopSpeech();
    spokenText = "";
    el("answer").textContent = "";
    el("badge").hidden = true;
    el("quote").hidden = true;
    el("choices").replaceChildren();
    if (frozen) context.putImageData(frozen, 0, 0);
    controls();
  }

  function controls() {
    el("send").disabled = !frozen || !el("utterance").value.trim() || Boolean(pending);
    el("speak").disabled = !spokenText || Boolean(pending) || Boolean(speaking);
    el("replay").disabled = !sample || el("utterance").value.trim() !== sentence || Boolean(pending);
  }

  function stopSpeech() {
    if (!speaking) return;
    speaking.controller.abort();
    speaking.audio?.pause();
    if (speaking.url) URL.revokeObjectURL(speaking.url);
    speaking = null;
  }

  function saveSnapshot(description, isSample) {
    frozen = context.getImageData(0, 0, canvas.width, canvas.height);
    source = description;
    sample = isSample;
    el("source").textContent = source;
    el("image").hidden = false;
    el("status").textContent = "Snapshot ready. Review the sentence, then explain it.";
    controls();
  }

  function highlight(objects) {
    context.putImageData(frozen, 0, 0);
    objects.forEach((object, index) => {
      const [top, left, bottom, right] = object.box;
      const x = left * canvas.width / 1000;
      const y = top * canvas.height / 1000;
      context.strokeStyle = "#71f0dc";
      context.lineWidth = 4;
      context.strokeRect(x, y, (right - left) * canvas.width / 1000, (bottom - top) * canvas.height / 1000);
      context.fillStyle = "#102a30";
      context.fillRect(x, y, 30, 28);
      context.fillStyle = "#ffffff";
      context.font = "bold 18px sans-serif";
      context.fillText(String(index + 1), x + 8, y + 21);
    });
  }

  function show(result, quote, scripted) {
    el("quote").textContent = `“${quote}”`;
    el("quote").hidden = false;
    el("answer").textContent = result.explanation;
    el("badge").textContent = scripted
      ? "Scripted demo"
      : { found: "Found", ambiguous: "Ambiguous", not_found: "Not found" }[result.status];
    el("badge").setAttribute("data-status", scripted ? "found" : result.status);
    el("badge").hidden = false;
    spokenText = result.explanation;
    el("status").textContent = scripted
      ? "Scripted CPU demo · no Gemini request or live audio"
      : result.status === "ambiguous"
        ? "Which object did they mean? Choose a candidate or clarify the sentence."
        : result.status === "not_found"
          ? "No matching object found. Try another snapshot."
          : "Gemini’s suggested reference · check it against the image";
    highlight(result.objects);
    controls();
    if (el("auto").checked) speak();
    el("choices").replaceChildren();
    result.objects.forEach((object, index) => {
      const item = document.createElement(result.status === "ambiguous" ? "button" : "span");
      item.textContent = `${index + 1}. ${object.label}`;
      item.className = "text-button";
      if (result.status === "ambiguous") {
        item.addEventListener("click", () => {
          highlight([object]);
          el("status").textContent = `You chose ${object.label}. Clarify the sentence for a more specific explanation.`;
          Array.from(el("choices").children).forEach((button) => button.setAttribute("aria-pressed", String(button === item)));
        });
      }
      el("choices").append(item);
    });
  }

  el("sample").addEventListener("click", () => {
    clearResult();
    canvas.width = 800;
    canvas.height = 450;
    context.fillStyle = "#17232e";
    context.fillRect(0, 0, 800, 450);
    context.fillStyle = "#e1f1f7";
    context.font = "24px sans-serif";
    context.fillText("Sample workbench · illustrated CPU demo", 30, 45);
    context.fillStyle = "#405568";
    context.fillRect(470, 120, 260, 230);
    context.fillStyle = "#080f17";
    context.fillRect(490, 175, 60, 35);
    context.fillRect(580, 175, 60, 35);
    context.fillStyle = "#b2c5d2";
    context.font = "20px sans-serif";
    context.fillText("PORT A", 487, 245);
    context.fillText("PORT B", 580, 245);
    context.lineWidth = 16;
    context.strokeStyle = "#438aff";
    context.beginPath();
    context.moveTo(80, 140);
    context.bezierCurveTo(400, 90, 90, 330, 400, 260);
    context.stroke();
    context.strokeStyle = "#ef906c";
    context.beginPath();
    context.moveTo(80, 330);
    context.bezierCurveTo(250, 430, 290, 320, 410, 375);
    context.stroke();
    el("utterance").value = sentence;
    saveSnapshot("Illustrated sample scene · scripted sentence · no devices opened", true);
  });

  el("freeze").addEventListener("click", () => {
    if (typeof snapshot === "undefined" || !snapshot?.session.selected_id || !frameFresh()) {
      el("status").textContent = "Start a camera session and select a visible person first, or load the CPU demo.";
      return;
    }
    const selected = snapshot.tracks.find((track) => track.track_id === snapshot.session.selected_id && track.visible);
    if (!selected) {
      el("status").textContent = "The selected person is out of view. Select again.";
      return;
    }
    const image = document.getElementById("camera-frame");
    if (!image.complete || !image.naturalWidth) return;
    clearResult();
    const scale = Math.min(1, 960 / image.naturalWidth);
    canvas.width = Math.round(image.naturalWidth * scale);
    canvas.height = Math.round(image.naturalHeight * scale);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    el("utterance").value = "";
    saveSnapshot(`Frozen scene · ${selected.person} selected · ${new Date().toLocaleTimeString()} · sentence entered manually`, false);
  });

  el("utterance").addEventListener("input", () => {
    clearResult();
    el("status").textContent = "Sentence changed. Explain again to update the highlight.";
  });

  el("send").addEventListener("click", async () => {
    if (!frozen || !el("utterance").value.trim() || pending) return;
    clearResult();
    const epoch = revision;
    const quote = el("utterance").value.trim();
    const controller = new AbortController();
    pending = controller;
    const timeout = setTimeout(() => controller.abort(), 30000);
    el("status").textContent = "Looking at this snapshot…";
    controls();
    try {
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: {"Content-Type": "application/json", "X-OneVoice-UI": "1"},
        body: JSON.stringify({image: canvas.toDataURL("image/jpeg", 0.85).split(",")[1], utterance: quote}),
        signal: controller.signal,
      });
      const result = await response.json();
      if (revision !== epoch) return;
      if (!response.ok) throw new Error(result.error || "Explanation failed.");
      show(result, quote, false);
    } catch (error) {
      if (revision === epoch) el("status").textContent = error.name === "AbortError"
        ? "Request timed out. Please retry."
        : error.message;
    } finally {
      clearTimeout(timeout);
      if (revision === epoch) { pending = null; controls(); }
    }
  });

  async function speak() {
    if (!spokenText || speaking) return;
    const epoch = revision;
    const controller = new AbortController();
    const current = { controller, audio: null, url: null };
    speaking = current;
    el("status").textContent = "Preparing speech…";
    controls();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch("/api/speak", {
        method: "POST",
        headers: {"Content-Type": "application/json", "X-OneVoice-UI": "1"},
        body: JSON.stringify({text: spokenText}),
        signal: controller.signal,
      });
      const result = await response.json();
      if (revision !== epoch || speaking !== current) return;
      if (!response.ok) throw new Error(result.error || "Speech failed.");
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
        el("status").textContent = error.name === "AbortError"
          ? "Speech timed out. Please retry."
          : error.message;
        controls();
      }
    } finally {
      clearTimeout(timeout);
    }
  }

  el("speak").addEventListener("click", speak);

  el("replay").addEventListener("click", () => {
    if (!sample || el("utterance").value.trim() !== sentence) return;
    clearResult();
    show({status: "found", explanation: "The highlighted blue cable is on the left of the workbench. This image does not establish which port it belongs in.", objects: [{label: "Blue cable", box: [240, 80, 660, 530]}]}, sentence, true);
  });

  el("clear").addEventListener("click", () => {
    clearResult();
    frozen = null;
    sample = false;
    canvas.width = 1;
    canvas.height = 1;
    el("image").hidden = true;
    el("utterance").value = "";
    el("source").textContent = "No snapshot yet.";
    el("status").textContent = "Snapshot and explanation cleared.";
    controls();
  });
  window.addEventListener("pagehide", clearResult);
})();
