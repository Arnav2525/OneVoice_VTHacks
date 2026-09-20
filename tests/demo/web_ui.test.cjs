"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const uiRoot = path.resolve(__dirname, "../../demo/web_ui");
const source = fs.readFileSync(path.join(uiRoot, "app.js"), "utf8");
const html = fs.readFileSync(path.join(uiRoot, "index.html"), "utf8");
const flush = () => new Promise((resolve) => setImmediate(resolve));

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase();
    this.children = [];
    this.listeners = new Map();
    this.attributes = new Map();
    this.style = {};
    this.dataset = {};
    this.hidden = false;
    this.disabled = false;
    this.textContent = "";
    this.className = "";
    const classes = new Set();
    this.classList = {
      add: (name) => classes.add(name),
      remove: (name) => classes.delete(name),
      contains: (name) => classes.has(name),
      toggle: (name, force = !classes.has(name)) => {
        if (force) classes.add(name); else classes.delete(name);
        return force;
      },
    };
  }

  append(...nodes) { for (const node of nodes) { node.parent = this; this.children.push(node); } }
  remove() { if (this.parent) this.parent.children = this.parent.children.filter((child) => child !== this); }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name) ?? null; }
  removeAttribute(name) { this.attributes.delete(name); }
  addEventListener(name, fn) { const list = this.listeners.get(name) || []; list.push(fn); this.listeners.set(name, list); }
  dispatch(name, event = {}) { return Promise.all((this.listeners.get(name) || []).map((fn) => fn({ target: this, ...event }))); }
  click() { if (!this.disabled) return this.dispatch("click"); }
  focus() { this.focused = true; }
  contains(node) { return node === this || this.children.some((child) => child.contains(node)); }
  getBoundingClientRect() { return { left: 0, top: 0, width: 640, height: 480 }; }
  get firstElementChild() { if (!this.children.length) this.append(new Element("span")); return this.children[0]; }
  get lastElementChild() { return this.children.at(-1) || this.firstElementChild; }
  get lastChild() { return this.lastElementChild; }
  closest(selector) {
    const matches = selector.split(",").some((part) => part === this.tagName.toLowerCase() || (part === "[contenteditable]" && this.attributes.has("contenteditable")));
    return matches ? this : this.parent?.closest(selector) || null;
  }
  querySelector(selector) {
    const descendants = this.children.flatMap((child) => [child, ...child.descendants()]);
    if (selector === ".target-label span") return descendants.find((node) => node.className === "target-label")?.children.find((node) => node.tagName === "SPAN");
    return descendants.find((node) => node.tagName.toLowerCase() === selector) || null;
  }
  descendants() { return this.children.flatMap((child) => [child, ...child.descendants()]); }
}

function payload(phase = "listening", overrides = {}) {
  const active = !["ready", "stopped", "error"].includes(phase);
  return {
    session: {
      phase, title: phase === "isolating" ? "Hearing Person 1" : "Select someone to hear",
      detail: "Session status", mode: "live", active, selected_id: "person-1",
      person: "Person 1", input_level: .2, output_level: .15, elapsed_s: 10, face_count: 1,
    },
    tracks: [{ track_id: "person-1", person: "Person 1", visible: true, bounding_box: [20, 30, 100, 150] }],
    recording: { active: false, elapsed_s: 0, path: null, error: null },
    record_busy: false, busy: active, synthetic: false,
    frame: { available: active, width: 640, height: 480, timestamp_ms: 1000 },
    ...overrides,
  };
}

function harness(extra = {}) {
  const elements = new Map();
  for (const match of html.matchAll(/<([a-z][\w-]*)\b([^>]*\bid="([^"]+)"[^>]*)>/g)) {
    const element = new Element(match[1]);
    element.hidden = /\bhidden\b/.test(match[2]);
    element.disabled = /\bdisabled\b/.test(match[2]);
    elements.set(match[3], element);
  }
  const document = new Element("document");
  document.body = new Element("body");
  document.documentElement = new Element("html");
  document.getElementById = (id) => { assert.ok(elements.has(id), `Missing HTML element: ${id}`); return elements.get(id); };
  document.createElement = (tag) => new Element(tag);
  document.createElementNS = (_, tag) => new Element(tag);
  const window = new Element("window");
  const timers = new Map();
  const requests = [];
  const images = [];
  const revoked = [];
  let nextTimer = 0;
  let nextUrl = 0;
  let now = 100;
  let decodeImmediately = true;
  const context = vm.createContext({
    document, window, console,
    performance: { now: () => now },
    ...extra,
    localStorage: { getItem: () => null, setItem() {} },
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    ResizeObserver: class { observe() {} },
    AbortSignal: { timeout: () => undefined },
    setTimeout(fn, delay) { timers.set(++nextTimer, { fn, delay }); return nextTimer; },
    clearTimeout(id) { timers.delete(id); },
    fetch(url, options) { const item = { url, options, ...deferred() }; requests.push(item); return item.promise; },
    URL: { createObjectURL: () => `blob:test-${++nextUrl}`, revokeObjectURL: (url) => revoked.push(url) },
    Image: class {
      set src(value) { this.url = value; images.push(this); if (decodeImmediately) queueMicrotask(() => this.onload()); }
    },
  });
  vm.runInContext(source, context, { filename: "app.js" });
  const evaluate = (code) => vm.runInContext(code, context);
  return {
    document, window, elements, timers, requests, images, revoked, evaluate,
    get: (id) => elements.get(id),
    advance(ms) { now += ms; },
    deferDecode() { decodeImmediately = false; },
    respond(request, data, { status = 200, timestamp = 1000 } = {}) {
      request.resolve({ ok: status >= 200 && status < 300, status,
        json: async () => data, blob: async () => ({}), headers: { get: () => String(timestamp) } });
    },
    latest(url) { const request = requests.findLast((item) => item.url === url); assert.ok(request, `No request to ${url}`); return request; },
    async boot(data = payload()) { this.respond(requests[0], data); await flush(); },
    takeTimer(name) {
      const entry = [...timers].find(([, timer]) => timer.fn.name === name);
      assert.ok(entry, `No ${name} timer`);
      timers.delete(entry[0]);
      return entry[1].fn();
    },
  };
}

test("late state response after Quit cannot restore hearing or output levels", async () => {
  const h = harness();
  await h.boot(payload("isolating"));
  const pendingPoll = h.takeTimer("poll");
  const request = h.latest("/api/state");
  const quitting = h.evaluate("quit()");
  h.respond(h.latest("/api/quit"), { ok: true });
  await quitting;
  h.respond(request, payload("isolating"));
  await pendingPoll;
  assert.equal(h.document.body.dataset.phase, "closed");
  assert.equal(h.get("connection").dataset.connected, "false");
  assert.equal(h.get("output-meter").getAttribute("aria-valuenow"), "0");
  assert.equal(h.get("session-button").disabled, true);
  assert.equal(h.timers.size, 0);
});

test("story handoff opens the camera-off stage without starting devices", async () => {
  const h = harness();
  await h.boot(payload("ready"));
  assert.equal(h.get("camera-view").hidden, false);
  assert.equal(h.get("video-surface").hidden, true);
  assert.equal(h.get("camera-wait").hidden, false);
  assert.match(h.get("camera-wait-text").textContent, /Camera is off/);
  assert.equal(h.requests.filter((request) => request.url === "/api/action").length, 0);
});

test("face matching explains the three-person limit and clears its hint on Stop", async () => {
  const h = harness();
  const identity = { enabled: true, registered: 3, capacity: 3, unmatched: 1, error: false };
  await h.boot(payload("listening", { identity }));
  assert.match(h.get("caption-detail").textContent, /Unrecognized faces stay unlabelled/);
  assert.match(h.get("people-count").lastElementChild.textContent, /recognized/);
  h.evaluate(`accept(${JSON.stringify(payload("listening", { identity: { ...identity, unmatched: 0 } }))})`);
  assert.equal(h.get("caption-detail").textContent, "");
  h.evaluate(`accept(${JSON.stringify(payload("stopped", { identity: {} }))})`);
  assert.equal(h.get("caption-detail").textContent, "");
});

test("Back/forward restoration resumes polling once and rejects the pre-navigation response", async () => {
  const h = harness();
  await h.boot(payload("isolating", { synthetic: true }));
  const oldPoll = h.takeTimer("poll");
  const oldRequest = h.latest("/api/state");
  await h.window.dispatch("pagehide");
  assert.equal(h.document.body.dataset.phase, "disconnected");
  assert.equal(h.get("output-meter").getAttribute("aria-valuenow"), "0");
  assert.equal(h.timers.size, 0);
  await h.window.dispatch("pageshow", { persisted: true });
  const restoredRequest = h.latest("/api/state");
  assert.notEqual(restoredRequest, oldRequest);
  h.respond(oldRequest, payload("isolating"));
  await oldPoll;
  assert.equal(h.document.body.dataset.phase, "disconnected");
  h.respond(restoredRequest, payload("stopped"));
  await flush();
  assert.equal(h.document.body.dataset.phase, "stopped");
  assert.equal([...h.timers.values()].filter((timer) => timer.fn.name === "poll").length, 1);
  assert.equal([...h.timers.values()].filter((timer) => timer.fn.name === "frames").length, 1);
});

test("repeated JPEG timestamp cannot keep an expired camera image or target clickable", async () => {
  const h = harness();
  await h.boot();
  let pendingFrame = h.takeTimer("frames");
  h.respond(h.latest("/api/frame"), null, { timestamp: 1000 });
  await pendingFrame;
  assert.equal(h.get("camera-frame").hidden, false);
  assert.equal(h.get("targets").children.length, 1);
  h.advance(2100);
  pendingFrame = h.takeTimer("frames");
  h.respond(h.latest("/api/frame"), null, { timestamp: 1000 });
  await pendingFrame;
  assert.equal(h.images.length, 1, "A duplicate capture must not be decoded as a new image");
  assert.equal(h.get("camera-frame").hidden, true);
  assert.equal(h.get("camera-wait").hidden, false);
  assert.equal(h.get("targets").children.length, 0);
  h.evaluate("render()");
  assert.equal(h.get("camera-frame").hidden, true, "State polling must not unhide stale pixels");
  pendingFrame = h.takeTimer("frames");
  h.respond(h.latest("/api/frame"), null, { timestamp: 1100 });
  await pendingFrame;
  assert.equal(h.get("camera-frame").hidden, false);
  assert.equal(h.get("targets").children.length, 1);
});

test("Stop session reports recording finalization before saved success", async () => {
  const h = harness();
  const recording = { active: false, elapsed_s: 12, path: "runs/sessions/clip", error: null };
  await h.boot(payload("stopping", { recording, record_busy: false }));
  assert.match(h.get("record-status-text").textContent, /Finishing/);
  assert.doesNotMatch(h.get("record-status-text").textContent, /saved/i);
  assert.equal(h.get("record-button").disabled, true);
  h.evaluate(`accept(${JSON.stringify(payload("stopped", { recording }))})`);
  assert.match(h.get("record-status-text").textContent, /saved/i);
});

test("Record and Clear shortcuts work with a button focused, but typing does not trigger commands", async () => {
  const h = harness();
  await h.boot(payload("listening", { synthetic: true }));
  const key = (value, target) => h.document.dispatch("keydown", { key: value, target, preventDefault() {} });
  await key("r", h.get("session-button"));
  let request = h.latest("/api/action");
  assert.equal(JSON.parse(request.options.body).action, "record");
  h.respond(request, payload("listening", { synthetic: true, recording: { active: true, elapsed_s: 0, path: "clip", error: null } }));
  await flush();
  await key("c", h.get("record-button"));
  request = h.latest("/api/action");
  assert.equal(JSON.parse(request.options.body).action, "clear");
  h.respond(request, payload());
  await flush();
  const count = h.requests.length;
  await key("r", h.document.createElement("input"));
  await key(" ", h.get("session-button"));
  assert.equal(h.requests.length, count, "Typing and native button Space must not dispatch a shortcut");
});

test("selection during JPEG decode discards the old image without stopping frame polling", async () => {
  const h = harness();
  await h.boot();
  h.deferDecode();
  const pendingFrame = h.takeTimer("frames");
  h.respond(h.latest("/api/frame"), null);
  await flush();
  assert.equal(h.images.length, 1);
  const selecting = h.evaluate('action("select", "person-1")');
  h.respond(h.latest("/api/action"), payload("focusing"));
  await selecting;
  h.images[0].onload();
  await pendingFrame;
  assert.equal(h.get("camera-frame").hidden, true);
  assert.ok(h.revoked.includes(h.images[0].url));
  assert.equal([...h.timers.values()].filter((timer) => timer.fn.name === "frames").length, 1);
});

test("an old command cannot overwrite restored state or unlock a newer command", async () => {
  const h = harness();
  await h.boot(payload("listening", { synthetic: true }));
  const oldAction = h.evaluate('action("select", "person-1")');
  const oldRequest = h.latest("/api/action");
  await h.window.dispatch("pagehide");
  await h.window.dispatch("pageshow", { persisted: true });
  h.respond(h.latest("/api/state"), payload("listening", { synthetic: true }));
  await flush();
  const newAction = h.evaluate('action("stop")');
  const newRequest = h.latest("/api/action");
  h.respond(oldRequest, payload("isolating", { synthetic: true }));
  await oldAction;
  assert.equal(h.document.body.dataset.phase, "listening");
  assert.equal(h.get("session-button").disabled, true, "The new command still owns the pending state");
  h.respond(newRequest, payload("stopping", { synthetic: true }));
  await newAction;
  assert.equal(h.document.body.dataset.phase, "stopping");
});

test("a JPEG decode spanning navigation cannot create a second frame polling loop", async () => {
  const h = harness();
  await h.boot();
  h.deferDecode();
  const oldFrame = h.takeTimer("frames");
  h.respond(h.latest("/api/frame"), null);
  await flush();
  await h.window.dispatch("pagehide");
  await h.window.dispatch("pageshow", { persisted: true });
  h.respond(h.latest("/api/state"), payload());
  await flush();
  h.images[0].onload();
  await oldFrame;
  assert.equal(h.get("camera-frame").hidden, true);
  assert.ok(h.revoked.includes(h.images[0].url));
  assert.equal([...h.timers.values()].filter((timer) => timer.fn.name === "frames").length, 1);
});

async function bootWithFrame(h, data) {
  await h.boot(data);
  const pendingFrame = h.takeTimer("frames");
  h.respond(h.latest("/api/frame"), null);
  await flush();
  await pendingFrame;
  h.evaluate(`accept(${JSON.stringify(data)})`);
}

const bubbleCaptions = (overrides = {}) => ({
  enabled: true,
  status: "ready",
  error: null,
  current: { track_id: "person-1", text: "Hello there", timestamp_ms: 5 },
  ...overrides,
});

test("live caption strip shows the selected person's words without face positioning", async () => {
  const h = harness();
  await bootWithFrame(h, payload("isolating", { captions: bubbleCaptions() }));
  const bubble = h.get("live-caption");
  assert.equal(bubble.hidden, false);
  assert.equal(bubble.classList.contains("visible"), true);
  assert.equal(h.get("live-caption-text").textContent, "Hello there");
  assert.equal(bubble.style.left, undefined);
  assert.equal(bubble.style.top, undefined);
});

test("caption strip excludes other speakers and preview scenes", async () => {
  const other = harness();
  await other.boot(
    payload("isolating", {
      captions: bubbleCaptions({ current: { track_id: "person-2", text: "Not selected", timestamp_ms: 5 } }),
    }),
  );
  assert.equal(other.get("live-caption-text").textContent, "Listening…");

  const listening = harness();
  await listening.boot(payload("listening", { captions: bubbleCaptions() }));
  assert.notEqual(listening.get("live-caption-text").textContent, "Hello there");

  const synthetic = harness();
  await synthetic.boot(payload("isolating", { synthetic: true, captions: bubbleCaptions() }));
  assert.equal(synthetic.get("live-caption").hidden, true);
});

test("caption strip returns to Listening after silence and shows the next line", async () => {
  const h = harness();
  await h.boot(payload("isolating", { captions: bubbleCaptions() }));
  assert.equal(h.get("live-caption").hidden, false);
  h.advance(9000);
  h.evaluate(`accept(${JSON.stringify(payload("isolating", { captions: bubbleCaptions() }))})`);
  assert.equal(h.get("live-caption-text").textContent, "Listening…");
  const next = bubbleCaptions({ current: { track_id: "person-1", text: "Second line", timestamp_ms: 9 } });
  h.evaluate(`accept(${JSON.stringify(payload("isolating", { captions: next }))})`);
  assert.equal(h.get("live-caption").hidden, false);
  assert.equal(h.get("live-caption-text").textContent, "Second line");
});

test("caption strip does not follow face movement", async () => {
  const h = harness();
  h.get("live-caption").offsetWidth = 200;
  h.get("live-caption").offsetHeight = 40;
  const nearTop = payload("isolating", { captions: bubbleCaptions() });
  nearTop.tracks[0].bounding_box = [0, 10, 60, 150];
  await bootWithFrame(h, nearTop);
  const bubble = h.get("live-caption");
  assert.equal(bubble.style.top, undefined);
  assert.equal(bubble.style.left, undefined);

  const nearRight = payload("isolating", { captions: bubbleCaptions() });
  nearRight.tracks[0].bounding_box = [600, 200, 40, 100];
  h.evaluate(`accept(${JSON.stringify(nearRight)})`);
  assert.equal(bubble.style.left, undefined);
  assert.equal(bubble.style.top, undefined);
});

test("caption strip displays connection errors and partial text updates", async () => {
  const h = harness();
  await h.boot(payload("isolating", { captions: bubbleCaptions({
    current: null, status: "error", error: "Check your API key",
  }) }));
  assert.equal(h.get("live-caption-text").textContent, "Check your API key");
  for (const [text, timestamp_ms] of [["Hello", 10], ["Hello world", 11]]) {
    const captions = bubbleCaptions({ current: {
      track_id: "person-1", text, timestamp_ms,
    } });
    h.evaluate(`accept(${JSON.stringify(payload("isolating", { captions }))})`);
    assert.equal(h.get("live-caption-text").textContent, text);
  }
});

test("before Start the stage shows a camera-off placeholder, not the old welcome page", async () => {
  const h = harness();
  await h.boot(payload("ready", { frame: { available: false } }));
  assert.equal(h.get("camera-view").hidden, false);
  assert.equal(h.get("video-surface").hidden, true);
  assert.equal(h.get("camera-wait").hidden, false);
  assert.equal(h.get("camera-wait-text").textContent, "Camera is off. Press Start listening.");
  assert.equal(h.get("focus-card").hidden, true);
  assert.doesNotMatch(html, /id="welcome"|arc_hero/);
});

test("once listening starts the live video surface replaces the placeholder", async () => {
  const h = harness();
  await h.boot(payload("listening"));
  assert.equal(h.get("video-surface").hidden, false);
  assert.equal(h.get("camera-wait-text").textContent, "Waiting for camera…");
  h.evaluate(`accept(${JSON.stringify(payload("stopped"))})`);
  assert.equal(h.get("video-surface").hidden, true);
  assert.equal(h.get("camera-wait-text").textContent, "Camera is off.");
});

test("arriving from the story page plays the reveal once and cleans the address", async () => {
  const replaced = [];
  const h = harness({
    location: { search: "?from=story", pathname: "/" },
    history: { replaceState: (...args) => replaced.push(args) },
    URLSearchParams,
  });
  assert.equal(h.document.body.classList.contains("from-story"), true);
  assert.deepEqual(replaced, [[null, "", "/"]]);
  const timer = [...h.timers.values()].find((entry) => entry.delay === 1600);
  assert.ok(timer);
  timer.fn();
  assert.equal(h.document.body.classList.contains("from-story"), false);
});

test("opening the app directly does not play the story reveal", async () => {
  const h = harness({
    location: { search: "", pathname: "/" },
    history: { replaceState() { assert.fail("address must not change"); } },
    URLSearchParams,
  });
  assert.equal(h.document.body.classList.contains("from-story"), false);
});

test("idle and routine states stay quiet: no filler copy, no empty recording status", async () => {
  const h = harness();
  await h.boot(payload("ready", { frame: { available: false } }));
  assert.equal(h.get("caption-kicker").textContent, "");
  assert.equal(h.get("caption-detail").textContent, "");
  assert.equal(h.get("record-status").hidden, true);
  assert.equal(h.get("status-detail").hidden, true);
  for (const filler of ["DESIGNED TO KEEP YOU", "Display preferences", "Live captions are off by default"]) {
    assert.ok(!html.includes(filler), filler);
  }
  h.evaluate(`accept(${JSON.stringify(payload("listening", { synthetic: true, recording: { active: true, elapsed_s: 3, path: "clip", error: null } }))})`);
  assert.equal(h.get("record-status").hidden, false);
  assert.match(h.get("record-status-text").textContent, /REC/);
});

test("problems still explain themselves: detail text shows only for error-like phases", async () => {
  const h = harness();
  await h.boot(payload("error", { session: { ...payload().session, phase: "error", active: false, title: "Session could not continue", detail: "The microphone could not be opened." } }));
  assert.equal(h.get("status-detail").hidden, false);
  assert.equal(h.get("status-detail").textContent, "The microphone could not be opened.");
});

test("the settings panel is gone and display preferences follow the operating system", async () => {
  assert.doesNotMatch(html, /id="settings"|id="reduce-motion"|id="high-contrast"|settings-toggle/);
  const h = harness();
  await h.boot(payload("ready", { frame: { available: false } }));
  assert.equal(h.document.body.dataset.preferencesReady, "true");
});

test("the camera stage takes the camera's own aspect ratio so the image fills the panel", async () => {
  const css = fs.readFileSync(path.join(uiRoot, "app.css"), "utf8");
  assert.match(css, /--cam-h: clamp\(360px, calc\(\(100cqw - 320px\) \/ var\(--cam-ar, 1\.7778\)\), 74vh\)/);
  assert.match(css, /main \{ container-type: inline-size; \}/);
});

test("the summary panel is a heading, one short disclosure and the controls, with no filler copy", () => {
  assert.doesNotMatch(html, /A short recap and key points|AFTER THE CONVERSATION|Live captions use ElevenLabs\. Speech playback/);
  assert.match(html, /<h2 id="summary-heading"[^>]*>Summary<\/h2>/);
  assert.match(html, /Stopping a recording sends its transcript to Gemini\./);
  assert.match(html, /id="summary-run"[^>]*title="Sends this session/);
  assert.match(html, /id="summary-speak"[^>]*title="Reads the summary aloud with ElevenLabs"/);
});
