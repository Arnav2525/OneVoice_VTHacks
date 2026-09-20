"use strict";

const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries(
  [
    "connection",
    "connection-label",
    "mode-label",
    "mode-context",
    "welcome",
    "camera-view",
    "camera-frame",
    "camera-wait",
    "video-surface",
    "sample-scene",
    "sample-people",
    "targets",
    "scene-notice",
    "focus-card",
    "phase-label",
    "session-time",
    "status-copy",
    "status-title",
    "status-detail",
    "input-label",
    "input-meter",
    "output-meter",
    "stage-hint",
    "people-count",
    "caption-kicker",
    "caption-detail",
    "session-button",
    "session-button-label",
    "session-icon",
    "record-button",
    "record-label",
    "clear-button",
    "captions-button",
    "live-caption",
    "live-caption-text",
    "record-status-text",
    "notice",
    "announcer",
    "settings",
    "settings-toggle",
    "settings-close",
    "reduce-motion",
    "high-contrast",
    "fullscreen",
    "quit-button",
  ].map((id) => [id, $(id)]),
);

let snapshot = null;
let connected = false;
let closed = false;
let suspended = false;
let commandPending = false;
let generation = 0;
let pageEpoch = 0;
let stateTimer;
let frameTimer;
let frameUrl = null;
let frameLoading = false;
let frameTimestamp = null;
let frameArrivedAt = 0;
let previousCopy = "";
let previousAnnouncement = "";
let actionError = "";
let recordingOperation = "";

const CAPTION_FADE_MS = 8000;
let lastCaptionKey = null;
let lastCaptionAt = 0;
const targetNodes = new Map();
const sampleNodes = new Map();
const recordable = new Set([
  "listening",
  "focusing",
  "isolating",
  "target_lost",
]);
const activePhases = new Set([
  "listening",
  "focusing",
  "isolating",
  "target_lost",
  "unavailable",
]);

function frameFresh() {
  return Boolean(
    connected &&
    !closed &&
    snapshot?.session.active &&
    frameUrl &&
    snapshot.frame.available &&
    performance.now() - frameArrivedAt < 2000 &&
    !["interrupted", "stopping"].includes(snapshot.session.phase),
  );
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.setAttribute("aria-hidden", "true");
  svg.append(use);
  return svg;
}

function elapsed(seconds) {
  const n = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${String(Math.floor(n / 60)).padStart(2, "0")}:${String(n % 60).padStart(2, "0")}`;
}

function meter(element, value) {
  const rms = Math.max(0, Math.min(1, Number(value) || 0));
  const amount = Math.max(0, (20 * Math.log10(Math.max(rms, 1e-6)) + 60) / 60);
  element.firstElementChild.style.transform = `scaleX(${amount})`;
  element.setAttribute("aria-valuenow", String(Math.round(amount * 100)));
}

function preferences() {
  for (const [name, media] of [
    ["reduce-motion", "(prefers-reduced-motion: reduce)"],
    ["high-contrast", "(prefers-contrast: more)"],
  ]) {
    let saved = null;
    try {
      saved = localStorage.getItem(`onevoice:${name}`);
    } catch {

    }
    const query = matchMedia(media);
    const update = (enabled) => {
      ui[name].checked = enabled;
      document.body.classList.toggle(name, enabled);
    };
    update(saved === null ? query.matches : saved === "true");
    query.addEventListener("change", (event) => {
      if (saved === null) update(event.matches);
    });
    ui[name].addEventListener("change", () => {
      saved = String(ui[name].checked);
      update(ui[name].checked);
      try {
        localStorage.setItem(`onevoice:${name}`, saved);
      } catch {

      }
    });
  }
  document.body.dataset.preferencesReady = "true";
}

function showSettings(open) {
  ui.settings.hidden = !open;
  ui["settings-toggle"].setAttribute("aria-expanded", String(open));
  if (open) ui["reduce-motion"].focus();
  else ui["settings-toggle"].focus();
}

function announce(message) {
  if (message === previousAnnouncement) return;
  previousAnnouncement = message;
  ui.announcer.textContent = message;
}

function createSample(track) {
  const person = document.createElement("div");
  person.className = "sample-person";
  person.setAttribute("aria-hidden", "true");
  const figure = document.createElement("div");
  figure.className = "person-figure";
  for (const name of ["neck", "shoulders", "head"]) {
    const part = document.createElement("div");
    part.className = name;
    figure.append(part);
  }
  person.append(figure);
  sampleNodes.set(track.track_id, person);
  ui["sample-people"].append(person);
}

function updateTargets() {
  if (!snapshot) return;
  const { session, synthetic, tracks } = snapshot;
  const visible = tracks.filter(
    (track) =>
      track.visible &&
      session.active &&
      connected &&
      !closed &&
      (synthetic || frameFresh()) &&
      activePhases.has(session.phase),
  );
  const ids = new Set(visible.map((track) => track.track_id));
  for (const [id, node] of targetNodes) {
    if (!ids.has(id)) {
      node.remove();
      targetNodes.delete(id);
    }
  }
  for (const [id, node] of sampleNodes) {
    if (!synthetic || !ids.has(id)) {
      node.remove();
      sampleNodes.delete(id);
    }
  }
  for (const track of visible) {
    if (synthetic && !sampleNodes.has(track.track_id)) createSample(track);
    let button = targetNodes.get(track.track_id);
    if (!button) {
      button = document.createElement("button");
      button.className = "target";
      const corners = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "svg",
      );
      corners.setAttribute("class", "reticle");
      corners.setAttribute("viewBox", "0 0 100 100");
      corners.setAttribute("preserveAspectRatio", "none");
      corners.setAttribute("aria-hidden", "true");
      const path = document.createElementNS(
        "http://www.w3.org/2000/svg",
        "path",
      );
      path.setAttribute(
        "d",
        "M20 1H8Q1 1 1 8V20 M80 1H92Q99 1 99 8V20 M1 80V92Q1 99 8 99H20 M80 99H92Q99 99 99 92V80",
      );
      path.setAttribute("vector-effect", "non-scaling-stroke");
      corners.append(path);
      button.append(corners);
      const label = document.createElement("span");
      label.className = "target-label";
      label.append(icon("people"), document.createElement("span"));
      button.append(label);
      button.addEventListener("click", () => action("select", track.track_id));
      targetNodes.set(track.track_id, button);
      ui.targets.append(button);
    }
    const selected = session.selected_id === track.track_id;
    button.classList.toggle("selected", selected);
    button.classList.toggle(
      "hearing",
      selected && session.phase === "isolating",
    );

    sampleNodes.get(track.track_id)?.classList.toggle("selected", selected);
    button.setAttribute("aria-label", `Select ${track.person}`);
    button.setAttribute("aria-pressed", String(selected));
    button.disabled = commandPending || !connected;
    button.querySelector(".target-label span").textContent = track.person;
    button
      .querySelector("use")
      .setAttribute("href", selected ? "#i-check" : "#i-people");
  }
  positionTargets();
}

function positionTargets() {
  if (!snapshot) return;
  const surface = ui["video-surface"].getBoundingClientRect();
  if (!surface.width || !surface.height) return;
  const { synthetic, tracks, frame } = snapshot;
  for (const track of tracks) {
    const button = targetNodes.get(track.track_id);
    if (!button) continue;
    let box;
    if (synthetic) {
      const person = sampleNodes.get(track.track_id)?.getBoundingClientRect();
      if (!person) continue;
      box = [
        person.left - surface.left,
        person.top - surface.top + person.height * 0.09,
        person.width,
        person.height * 0.93,
      ];
    } else {

      const width = frame.width || 640;
      const height = frame.height || 480;
      const scale = Math.min(surface.width / width, surface.height / height);
      const [x, y, w, h] = track.bounding_box;
      box = [
        (surface.width - width * scale) / 2 + x * scale,
        (surface.height - height * scale) / 2 + y * scale,
        w * scale,
        h * scale,
      ];
    }
    const [x, y, width, height] = box;
    Object.assign(button.style, {
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
    });

    if (track.track_id === snapshot.session.selected_id) {
      const bubble = ui["live-caption"];
      const half = (bubble.offsetWidth || 0) / 2;
      const tall = bubble.offsetHeight || 0;
      const centre = x + width / 2;
      Object.assign(bubble.style, {
        left: `${Math.min(Math.max(centre, half), Math.max(half, surface.width - half))}px`,
        top: `${tall ? Math.max(y, tall + 14) : y}px`,
      });
    }
  }
}

function render() {
  if (!snapshot) return;
  const {
    session,
    recording,
    synthetic,
    record_busy: recordBusy,
    busy,
  } = snapshot;
  const phase = closed ? "closed" : connected ? session.phase : "disconnected";
  document.body.dataset.phase = phase;
  document.body.dataset.active = String(session.active);
  document.body.dataset.synthetic = String(synthetic);
  document.body.dataset.recording = String(recording.active && connected);
  ui.connection.dataset.connected = String(connected);
  ui["connection-label"].textContent = closed
    ? "Session closed"
    : connected
      ? "Connected locally"
      : "Connection lost";
  ui["mode-label"].textContent = synthetic
    ? "CPU preview"
    : session.mode === "live"
      ? "Live listening"
      : "Camera preview";
  ui["mode-context"].textContent = synthetic
    ? "Simulated scene"
    : session.mode !== "live"
      ? "Isolation off"
      : "One voice at a time";

  ui.welcome.hidden = phase !== "ready";
  ui["camera-view"].hidden = !session.active;
  ui["sample-scene"].hidden = !synthetic;
  ui["scene-notice"].hidden = !synthetic;
  ui["camera-wait"].hidden = synthetic || frameFresh();
  ui["camera-frame"].hidden = synthetic || !frameFresh();
  ui["focus-card"].hidden = phase === "ready";
  ui["phase-label"].textContent =
    {
      isolating: "IN FOCUS",
      focusing: "FINDING YOUR VOICE",
      target_lost: "OUT OF VIEW",
      unavailable: "ISOLATION UNAVAILABLE",
      disconnected: "CONNECTION LOST",
      closed: "SESSION CLOSED",
    }[phase] || phase.replaceAll("_", " ").toUpperCase();
  ui["session-time"].textContent = elapsed(session.elapsed_s);
  let title = session.title;
  let detail = session.detail;
  if (phase === "disconnected") {
    title = "Connection interrupted";
    detail =
      "The local app is not responding. Audio status is unavailable; reconnecting…";
  }
  if (phase === "closed") {
    title = "Closing the application";
    detail =
      "You can close this tab. Check the terminal if shutdown reports a problem.";
  }
  if (synthetic && phase === "listening")
    detail =
      "Try selecting a person. This preview tests the controls; no audio is played.";
  if (phase === "listening" && !synthetic && session.mode !== "live")
    detail =
      "Your camera and microphone are active. Voice isolation is off in this preview.";
  if (phase === "isolating")
    detail = "Processed audio for your selected person is reaching playback.";
  const copy = `${phase}|${title}|${detail}`;
  if (copy !== previousCopy) {
    ui["status-title"].textContent = title;
    ui["status-detail"].textContent = detail;
    ui["status-copy"].classList.remove("changing");
    void ui["status-copy"].offsetWidth;
    ui["status-copy"].classList.add("changing");
    previousCopy = copy;
    announce(`${title}. ${detail}`);
  }
  ui["input-label"].textContent = synthetic ? "Sample input" : "Microphone";
  meter(ui["input-meter"], connected ? session.input_level : 0);
  meter(ui["output-meter"], phase === "isolating" ? session.output_level : 0);
  ui["stage-hint"].lastChild.textContent =
    ` ${phase === "ready" ? "Ready when you are" : title}`;
  ui["people-count"].lastElementChild.textContent = session.active
    ? `${connected ? session.face_count : "—"} people ${snapshot.identity?.enabled ? "recognized" : "visible"}`
    : "One voice at a time";
  ui["caption-kicker"].textContent = synthetic
    ? session.active
      ? "EXPLORE THE INTERACTION"
      : "LET’S GET STARTED"
    : session.active
      ? "YOUR LISTENING SESSION"
      : "READY TO CONNECT";
  ui["caption-detail"].textContent = session.active
    ? synthetic
      ? "Synthetic people. Isolation is off."
      : "Choose a face to change your focus."
    : "Your devices stay off until you start.";
  if (!synthetic && session.active && snapshot.identity?.enabled) {
    const identity = snapshot.identity;
    ui["caption-detail"].textContent = identity.error
      ? "Face matching paused. Move into good light and face the camera."
      : identity.unmatched
        ? identity.registered >= identity.capacity
          ? "Matching Person 1–3. Unrecognized faces stay unlabelled. Stop/start to reset people."
          : "Matching faces… Face the camera briefly to get a person label."
        : "Person 1–3 · Faces matched locally for this session. Stop clears face profiles.";
  }
  const startLabel = busy
    ? phase === "stopping"
      ? "Stopping…"
      : phase === "starting"
        ? "Cancel start"
        : "Stop session"
    : "Start listening";
  ui["session-button-label"].textContent = startLabel;
  ui["session-icon"].setAttribute("href", busy ? "#i-stop" : "#i-play");
  ui["session-button"].disabled =
    !connected || closed || commandPending || phase === "stopping";
  ui["record-button"].disabled =
    !connected ||
    closed ||
    commandPending ||
    recordBusy ||
    (!recording.active && !recordable.has(phase));
  ui["record-label"].textContent = recordBusy
    ? recordingOperation === "save"
      ? "Saving…"
      : "Preparing…"
    : recording.active
      ? "Stop recording"
      : "Record clip";
  ui["clear-button"].disabled =
    !connected ||
    closed ||
    commandPending ||
    !session.active ||
    !session.selected_id;

  const captions = snapshot.captions || {
    enabled: false,
    status: "off",
    error: null,
    current: null,
  };
  ui["captions-button"].disabled =
    !connected ||
    closed ||
    commandPending ||
    !session.active ||
    synthetic ||
    session.mode !== "live";
  ui["captions-button"].setAttribute("aria-pressed", String(captions.enabled));
  ui["captions-button"].classList.toggle(
    "loading",
    captions.status === "loading",
  );
  ui["captions-button"].setAttribute(
    "aria-label",
    captions.status === "loading"
      ? "Loading live captions…"
      : captions.enabled
        ? "Disable live captions"
        : "Enable live captions",
  );
  const currentCaption = captions.current;
  const captionKey = currentCaption
    ? `${currentCaption.track_id}|${currentCaption.timestamp_ms}`
    : null;
  if (captionKey !== lastCaptionKey) {
    lastCaptionKey = captionKey;
    lastCaptionAt = performance.now();
    if (currentCaption)
      ui["live-caption-text"].textContent = currentCaption.text;
  }
  const captionFresh =
    Boolean(currentCaption) &&
    performance.now() - lastCaptionAt < CAPTION_FADE_MS;
  const showCaption =
    captionFresh &&
    !synthetic &&
    phase === "isolating" &&
    currentCaption.track_id === session.selected_id;
  ui["live-caption"].hidden = !showCaption;
  ui["live-caption"].classList.toggle("visible", showCaption);
  if (showCaption) positionTargets();
  ui["record-status-text"].textContent = closed
    ? "Closing the application…"
    : !connected
      ? "Recording status unavailable"
      : recording.error
        ? "Recording needs attention"
        : phase === "stopping" && recording.path
          ? "Finishing your clip…"
          : recordBusy
            ? recordingOperation === "save"
              ? "Saving your clip…"
              : "Preparing your clip…"
            : recording.active
              ? `REC ${elapsed(recording.elapsed_s)}`
              : recording.path
                ? "Clip saved on this laptop"
                : "Nothing is being saved";
  const notice =
    actionError ||
    recording.error ||
    (captions.status === "error"
      ? captions.error || "Live captions could not start."
      : "") ||
    (phase === "error" ? session.detail : "");
  ui.notice.textContent = notice;
  ui.notice.hidden = !notice;
  ui["quit-button"].disabled = closed || commandPending;
  ui["camera-view"].classList.toggle(
    "stale",
    !connected || phase === "interrupted",
  );
  updateTargets();
}

function accept(data) {
  if (closed || suspended) return;
  if (
    !data ||
    !data.session ||
    !Array.isArray(data.tracks) ||
    !data.recording ||
    !data.frame
  )
    throw new Error("Invalid response from the local app.");
  snapshot = data;
  connected = true;
  render();
}

async function poll() {
  if (closed || suspended || commandPending) return;
  const requestedGeneration = generation;
  try {
    const response = await fetch("/api/state", {
      cache: "no-store",
      signal: AbortSignal.timeout(2500),
    });
    if (!response.ok) throw new Error("Local app unavailable");
    const data = await response.json();
    if (
      requestedGeneration === generation &&
      !commandPending &&
      !closed &&
      !suspended
    )
      accept(data);
  } catch {
    if (
      requestedGeneration === generation &&
      !commandPending &&
      !closed &&
      !suspended
    ) {
      connected = false;
      ui.connection.dataset.connected = "false";
      ui["connection-label"].textContent = "Reconnecting…";
      render();
    }
  } finally {
    if (!closed && !suspended && requestedGeneration === generation)
      stateTimer = setTimeout(poll, 180);
  }
}

async function action(name, trackId) {
  if (!connected || closed || commandPending || !snapshot) return;
  commandPending = true;
  generation += 1;
  const requestGeneration = generation;
  actionError = "";
  if (name === "record")
    recordingOperation = snapshot.recording.active ? "save" : "prepare";
  render();
  try {
    const response = await fetch("/api/action", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-OneVoice-UI": "1" },
      body: JSON.stringify({
        action: name,
        ...(trackId ? { track_id: trackId } : {}),
      }),
      signal: AbortSignal.timeout(3500),
    });
    const data = await response.json();
    if (requestGeneration !== generation || closed || suspended) return;
    if (!response.ok)
      throw new Error(data.error || "That action could not be completed.");
    accept(data);
  } catch (error) {
    if (requestGeneration !== generation || closed || suspended) return;
    actionError =
      error.name === "TimeoutError"
        ? "The action timed out. Checking the session before you retry."
        : error.message;
    connected = false;
  } finally {
    if (requestGeneration === generation) {
      commandPending = false;
      render();
      clearTimeout(stateTimer);
      if (!closed && !suspended) stateTimer = setTimeout(poll, 0);
    }
  }
}

async function frames() {
  if (closed || suspended) return;
  const epoch = pageEpoch;
  try {
    if (
      snapshot?.session.active &&
      !snapshot.synthetic &&
      snapshot.frame.available &&
      connected &&
      !frameLoading
    ) {
      frameLoading = true;
      const requestedGeneration = generation;
      try {
        const response = await fetch("/api/frame", {
          cache: "no-store",
          signal: AbortSignal.timeout(2000),
        });
        const timestampHeader = response.headers.get("X-Frame-Timestamp-Ms");
        const stamp = Number(timestampHeader);
        if (
          response.status === 200 &&
          timestampHeader !== null &&
          Number.isFinite(stamp) &&
          stamp > (frameTimestamp ?? -1)
        ) {
          const blob = await response.blob();
          if (
            requestedGeneration === generation &&
            snapshot?.session.active &&
            !closed &&
            !suspended
          ) {
            const url = URL.createObjectURL(blob);
            const probe = new Image();
            await new Promise((resolve, reject) => {
              probe.onload = resolve;
              probe.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error("Invalid camera frame"));
              };
              probe.src = url;
            });
            if (
              requestedGeneration !== generation ||
              closed ||
              suspended ||
              !snapshot?.session.active
            ) {
              URL.revokeObjectURL(url);
              return;
            }
            const previous = frameUrl;
            frameUrl = url;
            frameTimestamp = stamp;
            frameArrivedAt = performance.now();
            ui["camera-frame"].src = url;
            if (previous) URL.revokeObjectURL(previous);
            render();
          }
        }
      } catch {

      } finally {
        frameLoading = false;
      }
    }
    if (!snapshot?.session.active || snapshot?.synthetic) {
      if (frameUrl) URL.revokeObjectURL(frameUrl);
      frameUrl = null;
      frameTimestamp = null;
      ui["camera-frame"].removeAttribute("src");
    } else if (frameUrl && !frameFresh()) {
      render();
    }
  } finally {
    if (!closed && !suspended && epoch === pageEpoch)
      frameTimer = setTimeout(frames, 100);
  }
}

async function fullscreen() {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch {
    actionError = "Fullscreen is unavailable in this browser window.";
    render();
  }
}

async function quit() {
  if (closed || commandPending) return;
  commandPending = true;
  generation += 1;
  const requestGeneration = generation;
  clearTimeout(stateTimer);
  render();
  try {
    const response = await fetch("/api/quit", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-OneVoice-UI": "1" },
      body: "{}",
      signal: AbortSignal.timeout(3000),
    });
    if (requestGeneration !== generation || suspended) return;
    if (!response.ok)
      throw new Error(
        "Could not close the app. Stop the session and close the terminal.",
      );
    closed = true;
    connected = false;
    clearTimeout(stateTimer);
    clearTimeout(frameTimer);
    if (frameUrl) URL.revokeObjectURL(frameUrl);
  } catch (error) {
    if (requestGeneration !== generation || suspended) return;
    actionError = error.message;
  } finally {
    if (requestGeneration === generation) {
      commandPending = false;
      render();
      if (!closed && !suspended) stateTimer = setTimeout(poll, 0);
    }
  }
}

ui["session-button"].addEventListener("click", () =>
  action(snapshot?.busy ? "stop" : "start"),
);
ui["record-button"].addEventListener("click", () => action("record"));
ui["clear-button"].addEventListener("click", () => action("clear"));
ui["captions-button"].addEventListener("click", () => action("captions"));
ui["settings-toggle"].addEventListener("click", () =>
  showSettings(ui.settings.hidden),
);
ui["settings-close"].addEventListener("click", () => showSettings(false));
ui.fullscreen.addEventListener("click", fullscreen);
ui["quit-button"].addEventListener("click", quit);
document.addEventListener("fullscreenchange", () => {
  ui.fullscreen.setAttribute(
    "aria-label",
    document.fullscreenElement ? "Exit fullscreen" : "Enter fullscreen",
  );
  positionTargets();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !ui.settings.hidden) {
    showSettings(false);
    return;
  }
  if (
    event.repeat ||
    event.ctrlKey ||
    event.altKey ||
    event.metaKey ||
    event.target.closest("input,textarea,select,[contenteditable]")
  )
    return;
  const key = event.key.toLowerCase();
  if (key === " " && !event.target.closest("button,a")) {
    event.preventDefault();
    if (!ui["session-button"].disabled) ui["session-button"].click();
  } else if (key === "r" && !ui["record-button"].disabled)
    ui["record-button"].click();
  else if (key === "c" && !ui["clear-button"].disabled)
    ui["clear-button"].click();
  else if (key === "f") fullscreen();
});
document.addEventListener("pointerdown", (event) => {
  if (
    !ui.settings.hidden &&
    !ui.settings.contains(event.target) &&
    !ui["settings-toggle"].contains(event.target)
  )
    showSettings(false);
});
new ResizeObserver(positionTargets).observe(ui["video-surface"]);
window.addEventListener("pagehide", () => {
  suspended = true;
  connected = false;
  generation += 1;
  pageEpoch += 1;
  clearTimeout(stateTimer);
  clearTimeout(frameTimer);
  if (frameUrl) URL.revokeObjectURL(frameUrl);
  frameUrl = null;
  frameTimestamp = null;
  frameArrivedAt = 0;
  render();
});
window.addEventListener("pageshow", (event) => {
  if (!event.persisted || closed) return;
  suspended = false;
  commandPending = false;
  generation += 1;
  pageEpoch += 1;
  clearTimeout(stateTimer);
  clearTimeout(frameTimer);
  render();
  poll();
  frames();
});
preferences();
poll();
frames();
