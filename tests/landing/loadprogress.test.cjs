"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const test = require("node:test");
const { pathToFileURL } = require("node:url");

const modulePath = pathToFileURL(path.resolve(__dirname, "../../one-voice-working-copy/loadprogress.js")).href;

function fakeDocument() {
  const nodes = new Map();
  for (const id of ["loading", "loading-lit", "loading-dim"]) {
    nodes.set(id, { id, textContent: "", attributes: new Map(), classList: { contains: () => false }, setAttribute(k, v) { this.attributes.set(k, v); } });
  }
  globalThis.document = { getElementById: (id) => nodes.get(id) ?? null };
  return nodes;
}

async function load() {
  const nodes = fakeDocument();
  const mod = await import(`${modulePath}?t=${Math.random()}`);
  mod.resetLoad();
  return { nodes, mod };
}

test("with nothing loaded the bar is all dashes", async () => {
  const { mod } = await load();
  const { text, lit } = mod.asciiBar(0);
  assert.equal(text.length, 26);
  assert.equal(lit, 0);
  assert.match(text, /^-{26}$/);
});

test("filled cells only ever show = or +, unfilled cells stay -, and a full bar has no dashes", async () => {
  const { mod } = await load();
  for (let step = 0; step < 12; step += 1) {
    const { text, lit } = mod.asciiBar(0.5, step);
    assert.equal(lit, 13);
    assert.match(text.slice(0, lit), /^[=+]+$/);
    assert.match(text.slice(lit), /^-+$/);
  }
  assert.match(mod.asciiBar(1, 3).text, /^[=+]{26}$/);
});

test("filled cells flicker between frames but the amount filled never changes", async () => {
  const { mod } = await load();
  const frames = new Set(Array.from({ length: 8 }, (_, step) => mod.asciiBar(0.6, step).text));
  assert.ok(frames.size > 1, "animation frames differ");
  assert.ok([...frames].every((text) => mod.asciiBar(0.6, 0).lit === [...text].filter((c) => c !== "-").length));
});

test("progress only moves forward, is clamped to 0 to 100 percent, and updates the page", async () => {
  const { nodes, mod } = await load();
  mod.setLoad(0.5);
  mod.setLoad(0.2);
  assert.equal(nodes.get("loading").attributes.get("aria-valuenow"), "50");
  assert.equal(nodes.get("loading-lit").textContent.length, 13);
  assert.equal(nodes.get("loading-dim").textContent.length, 13);
  mod.setLoad(7);
  assert.equal(nodes.get("loading").attributes.get("aria-valuenow"), "100");
  assert.equal(nodes.get("loading-dim").textContent, "");
  mod.setLoad(-3);
  assert.equal(nodes.get("loading").attributes.get("aria-valuenow"), "100");
});

test("a streamed download reports real progress between the two milestones and returns every byte", async () => {
  const { nodes, mod } = await load();
  const chunks = [new Uint8Array(250).fill(1), new Uint8Array(250).fill(2), new Uint8Array(500).fill(3)];
  const seen = [];
  const original = nodes.get("loading").setAttribute.bind(nodes.get("loading"));
  nodes.get("loading").setAttribute = (k, v) => { if (k === "aria-valuenow") seen.push(Number(v)); original(k, v); };
  let index = 0;
  const response = {
    headers: { get: (name) => ({ "X-Uncompressed-Length": "1000" })[name] ?? null },
    body: { getReader: () => ({ read: async () => (index < chunks.length ? { done: false, value: chunks[index++] } : { done: true }) }) },
  };
  const buffer = await mod.readWithProgress(response, 0.4, 0.86);
  const bytes = new Uint8Array(buffer);
  assert.equal(bytes.length, 1000);
  assert.deepEqual([bytes[0], bytes[300], bytes[999]], [1, 2, 3]);
  assert.ok(seen.length >= 3);
  assert.ok(seen.every((value, i) => i === 0 || value >= seen[i - 1]), "never goes backwards");
  assert.ok(seen.some((value) => value > 40 && value < 86), "reports in-between values");
  assert.equal(seen.at(-1), 86);
});

test("a response with no known size still finishes the stage", async () => {
  const { nodes, mod } = await load();
  const response = { headers: { get: () => null }, body: null, arrayBuffer: async () => new ArrayBuffer(8) };
  const buffer = await mod.readWithProgress(response, 0.4, 0.86);
  assert.equal(buffer.byteLength, 8);
  assert.equal(nodes.get("loading").attributes.get("aria-valuenow"), "86");
});

test("the loading manager reports a share of the range as each texture finishes", async () => {
  const { nodes, mod } = await load();
  const manager = {};
  mod.trackLoadingManager(manager, 0.05, 0.38);
  manager.onProgress("a.jpg", 2, 4);
  assert.equal(nodes.get("loading").attributes.get("aria-valuenow"), String(Math.round((0.05 + 0.33 * 0.5) * 100)));
  manager.onProgress("d.jpg", 4, 4);
  assert.equal(nodes.get("loading").attributes.get("aria-valuenow"), "38");
});
