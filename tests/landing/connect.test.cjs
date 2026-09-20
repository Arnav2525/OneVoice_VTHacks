"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const siteRoot = path.resolve(__dirname, "../../one-voice-working-copy");
const script = fs.readFileSync(path.join(siteRoot, "connect.js"), "utf8");
const html = fs.readFileSync(path.join(siteRoot, "index.html"), "utf8");
const flush = () => new Promise((resolve) => setImmediate(resolve));

function harness({ running = true, reducedMotion = false } = {}) {
  const listeners = new Map();
  const navigations = [];
  const timers = [];
  const appended = [];
  const fetched = [];
  let isRunning = running;
  const make = (extra = {}) => ({
    hidden: false,
    textContent: "",
    listeners: new Map(),
    addEventListener(name, fn) { this.listeners.set(name, fn); },
    fire(name) { return this.listeners.get(name)?.({ target: this }); },
    setAttribute() {},
    ...extra,
  });
  const dialog = make({ closed: 0, close() { this.closed += 1; } });
  const elements = {
    "connection-dialog": dialog,
    "connection-status": make(),
    "connection-help": make({ hidden: true }),
    "connection-retry": make(),
  };
  const document = {
    getElementById: (id) => elements[id] || null,
    addEventListener(name, fn) { listeners.set(name, fn); },
    createElement: () => make({ className: "" }),
    body: { append: (node) => appended.push(node) },
  };
  const context = vm.createContext({
    document,
    fetch: async (url, options) => {
      fetched.push({ url, options });
      if (!isRunning) throw new TypeError("Failed to fetch");
      return { ok: true };
    },
    AbortController,
    setTimeout: (fn, delay) => { timers.push({ fn, delay }); return timers.length; },
    clearTimeout() {},
    matchMedia: () => ({ matches: reducedMotion }),
    location: {
      set href(value) { navigations.push(value); },
      get href() { return navigations.at(-1) || ""; },
    },
  });
  vm.runInContext(script, context, { filename: "connect.js" });
  const trigger = { closest: (selector) => (selector === "[data-connect]" ? {} : null) };
  const other = { closest: () => null };
  return {
    elements, dialog, navigations, timers, appended, fetched,
    setRunning(value) { isRunning = value; },
    click(target = trigger) { listeners.get("click")({ target }); return flush(); },
  };
}

test("landing page has the connection status and help elements the script needs", () => {
  for (const id of ["connection-status", "connection-help", "connection-retry", "connection-dialog"]) {
    assert.match(html, new RegExp(`id="${id}"`), id);
  }
  assert.match(html, /connect\.js/);
  assert.match(html, /connect\.css/);
  assert.doesNotMatch(html, /Connection is not enabled/);
});

test("Connect pings the fixed local app port with a plain request", async () => {
  const h = harness();
  await h.click();
  assert.equal(h.fetched.length, 1);
  assert.equal(h.fetched[0].url, "http://127.0.0.1:8771/api/ping");
  assert.equal(h.fetched[0].options.mode, undefined);
});

test("when the app is running the iris plays, then the app opens with the story flag", async () => {
  const h = harness();
  await h.click();
  assert.match(h.elements["connection-status"].textContent, /Connected/);
  assert.equal(h.dialog.closed, 1);
  assert.equal(h.appended.length, 1);
  assert.equal(h.appended[0].className, "connect-iris");
  assert.deepEqual(h.navigations, []);
  h.appended[0].fire("animationend");
  assert.deepEqual(h.navigations, ["http://127.0.0.1:8771/?from=story"]);
});

test("if the animation never fires the fallback timer still opens the app", async () => {
  const h = harness();
  await h.click();
  const fallback = h.timers.find((timer) => timer.delay === 1600);
  assert.ok(fallback);
  fallback.fn();
  assert.deepEqual(h.navigations, ["http://127.0.0.1:8771/?from=story"]);
});

test("reduced motion skips the animation and goes straight to the app", async () => {
  const h = harness({ reducedMotion: true });
  await h.click();
  assert.equal(h.appended.length, 0);
  assert.deepEqual(h.navigations, ["http://127.0.0.1:8771/?from=story"]);
});

test("when the app is not running the dialog explains how to start it and can retry", async () => {
  const h = harness({ running: false });
  await h.click();
  assert.match(h.elements["connection-status"].textContent, /not running/);
  assert.equal(h.elements["connection-help"].hidden, false);
  assert.equal(h.navigations.length, 0);
  assert.equal(h.appended.length, 0);
  h.setRunning(true);
  await h.elements["connection-retry"].fire("click");
  await flush();
  assert.equal(h.elements["connection-help"].hidden, true);
  assert.equal(h.appended.length, 1);
});

test("clicks elsewhere on the page do nothing, and closing the dialog cancels a pending check", async () => {
  const h = harness();
  await h.click({ closest: () => null });
  assert.equal(h.fetched.length, 0);
  const pending = h.click();
  h.dialog.fire("close");
  await pending;
  assert.equal(h.appended.length, 0);
  assert.equal(h.navigations.length, 0);
});
