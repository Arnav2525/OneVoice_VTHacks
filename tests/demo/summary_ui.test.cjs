const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function harness(fetch = async () => { throw Error('Unexpected network'); }, extra = {}) {
  const elements = new Map();
  function element() {
    return {value: '', textContent: '', hidden: false, checked: false, children: [], listeners: {},
      addEventListener(name, fn) { this.listeners[name] = fn; },
      replaceChildren() { this.children = []; }, append(child) { this.children.push(child); },
      setAttribute(name, value) { this[name] = value; },
      fire(name = 'click') { return this.listeners[name](); }};
  }
  const html = fs.readFileSync(path.resolve(__dirname, '../../demo/web_ui/index.html'), 'utf8');
  for (const [, id] of html.matchAll(/id="(summary-[^"]+)"/g)) elements.set(id, element());
  const document = {getElementById: id => {assert.ok(elements.has(id), id); return elements.get(id);}, createElement: element};
  vm.runInNewContext(fs.readFileSync(path.resolve(__dirname, '../../demo/web_ui/summary.js'), 'utf8'), {
    document, window: {addEventListener() {}}, fetch, AbortController, setTimeout, clearTimeout, ...extra,
  });
  return name => elements.get('summary-' + name);
}

function speechKit(summaryReply) {
  const calls = [];
  const played = [];
  const fetch = async (url, options) => {
    calls.push({url, body: JSON.parse(options.body)});
    if (url === '/api/summarize') return summaryReply();
    return {ok: true, json: async () => ({audio: 'AAAA'})};
  };
  class FakeAudio {
    constructor(url) { this.url = url; }
    addEventListener() {}
    play() { played.push(this.url); return Promise.resolve(); }
    pause() {}
  }
  const extra = {
    atob: (text) => Buffer.from(text, 'base64').toString('binary'),
    Uint8Array, Blob, Audio: FakeAudio,
    URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}},
  };
  return {calls, played, extra, fetch};
}

const GOOD = () => ({ok: true, json: async () => ({summary: 'They agreed on lunch.', key_points: ['Noon', 'Cafe']})});

test('scripted demo shows a summary and never touches the network', () => {
  const el = harness();
  el('sample').fire();
  assert.match(el('text').textContent, /meet at noon/);
  assert.equal(el('points').children.length, 3);
  assert.equal(el('badge').textContent, 'Scripted demo');
  assert.match(el('status').textContent, /no Gemini request/);
  assert.equal(el('speak').disabled, false);
});

test('Summarize posts an empty body so the server uses its own transcript', async () => {
  const kit = speechKit(GOOD);
  const el = harness(kit.fetch, kit.extra);
  await el('run').fire();
  assert.equal(kit.calls.length, 1);
  assert.deepEqual(kit.calls[0], {url: '/api/summarize', body: {}});
  assert.equal(el('text').textContent, 'They agreed on lunch.');
  assert.deepEqual(el('points').children.map((item) => item.textContent), ['Noon', 'Cafe']);
  assert.equal(el('badge').textContent, 'Gemini summary');
  assert.equal(el('run').disabled, false);
});

test('A server error is shown and leaves Speak disabled', async () => {
  const kit = speechKit(() => ({ok: false, json: async () => ({error: 'No captions yet.'})}));
  const el = harness(kit.fetch, kit.extra);
  await el('run').fire();
  assert.match(el('status').textContent, /No captions yet/);
  assert.equal(el('text').textContent, '');
  assert.equal(el('speak').disabled, true);
  assert.equal(el('run').disabled, false);
});

test('Speak sends only the summary text to ElevenLabs', async () => {
  const kit = speechKit(GOOD);
  const el = harness(kit.fetch, kit.extra);
  await el('run').fire();
  await el('speak').fire();
  const speak = kit.calls.find((call) => call.url === '/api/speak');
  assert.deepEqual(speak.body, {text: 'They agreed on lunch.'});
  assert.deepEqual(kit.played, ['blob:test']);
  assert.match(el('status').textContent, /Speaking with ElevenLabs/);
});

test('Automatic speaking is opt-in: nothing is sent to ElevenLabs unless it is on', async () => {
  const kit = speechKit(GOOD);
  const el = harness(kit.fetch, kit.extra);
  await el('run').fire();
  await Promise.resolve();
  assert.equal(kit.calls.filter((call) => call.url === '/api/speak').length, 0);
});

test('With automatic speaking on, each new summary is spoken once', async () => {
  const kit = speechKit(GOOD);
  const el = harness(kit.fetch, kit.extra);
  el('auto').checked = true;
  await el('run').fire();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(kit.calls.filter((call) => call.url === '/api/speak').length, 1);
});

test('Clear removes the result, the badge and the ability to speak', async () => {
  const kit = speechKit(GOOD);
  const el = harness(kit.fetch, kit.extra);
  await el('run').fire();
  el('clear').fire();
  assert.equal(el('text').textContent, '');
  assert.equal(el('points').children.length, 0);
  assert.equal(el('badge').hidden, true);
  assert.equal(el('speak').disabled, true);
});

test('A late summary after Clear is discarded', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const kit = speechKit(async () => { await gate; return GOOD(); });
  const el = harness(kit.fetch, kit.extra);
  const running = el('run').fire();
  el('clear').fire();
  release();
  await running;
  assert.equal(el('text').textContent, '');
  assert.equal(el('speak').disabled, true);
});
