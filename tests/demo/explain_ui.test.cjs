const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function harness(fetch = async () => { throw Error('Unexpected network'); }, extra = {}) {
  const elements = new Map();
  const drawing = new Proxy({}, {get: (_, key) => key === 'getImageData' ? () => ({pixels: true}) : () => {}});
  function element() {
    return {value: '', textContent: '', hidden: false, children: [], listeners: {},
      addEventListener(name, fn) { this.listeners[name] = fn; },
      replaceChildren() { this.children = []; }, append(child) { this.children.push(child); },
      setAttribute() {}, getContext: () => drawing,
      toDataURL: () => 'data:image/jpeg;base64,TEST',
      fire(name = 'click') { return this.listeners[name](); }};
  }
  const html = fs.readFileSync(path.resolve(__dirname, '../../demo/web_ui/index.html'), 'utf8');
  for (const [, id] of html.matchAll(/id="(explain-[^"]+)"/g)) elements.set(id, element());
  const document = {getElementById: id => {assert.ok(elements.has(id), id); return elements.get(id);}, createElement: element};
  vm.runInNewContext(fs.readFileSync(path.resolve(__dirname, '../../demo/web_ui/explain.js'), 'utf8'), {
    document, window: {addEventListener() {}}, fetch, AbortController, setTimeout, clearTimeout, ...extra,
  });
  return name => elements.get('explain-' + name);
}

test('CPU sample is explicitly scripted and never calls provider', () => {
  const el = harness();
  el('sample').fire();
  el('replay').fire();
  assert.match(el('status').textContent, /Scripted CPU demo/);
  assert.match(el('answer').textContent, /blue cable/);
  assert.equal(el('choices').children.length, 1);
});

test('editing the sentence removes stale results and disables unrelated replay', () => {
  const el = harness();
  el('sample').fire(); el('replay').fire();
  el('utterance').value = 'Which port?'; el('utterance').fire('input');
  assert.equal(el('answer').textContent, '');
  assert.equal(el('replay').disabled, true);
  assert.equal(el('choices').children.length, 0);
});

test('clearing a snapshot discards a late provider response', async () => {
  let resolve;
  const response = new Promise(done => { resolve = done; });
  const el = harness(() => response);
  el('sample').fire();
  const pending = el('send').fire();
  el('clear').fire();
  resolve({ok: true, json: async () => ({status:'not_found', explanation:'Late result', objects:[]})});
  await pending;
  assert.equal(el('answer').textContent, '');
  assert.equal(el('image').hidden, true);
  assert.equal(el('send').disabled, true);
});

test('ambiguous results let the user choose without claiming model certainty', async () => {
  const el = harness(async () => ({ok:true, json: async () => ({status:'ambiguous', explanation:'Which port?', objects:[
    {label:'Port A', box:[10,10,20,20]}, {label:'Port B', box:[30,30,40,40]},
  ]})}));
  el('sample').fire(); await el('send').fire();
  assert.equal(el('choices').children.length, 2);
  el('choices').children[1].fire();
  assert.match(el('status').textContent, /You chose Port B/);
});

test('Speak stays disabled until an explanation exists and only sends that text', async () => {
  const calls = [];
  const played = [];
  const fetch = async (url, options) => {
    calls.push({url, body: JSON.parse(options.body)});
    return {ok: true, json: async () => ({audio: 'AAAA'})};
  };
  class FakeAudio {
    constructor(url) { this.url = url; this.listeners = {}; }
    addEventListener(name, fn) { this.listeners[name] = fn; }
    play() { played.push(this.url); return Promise.resolve(); }
    pause() {}
  }
  const el = harness(fetch, {
    atob: (text) => Buffer.from(text, 'base64').toString('binary'),
    Uint8Array, Blob, Audio: FakeAudio,
    URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}},
  });
  el('sample').fire();
  assert.equal(el('speak').disabled, true);
  el('replay').fire();
  assert.equal(el('speak').disabled, false);
  await el('speak').fire();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/speak');
  assert.deepEqual(Object.keys(calls[0].body), ['text']);
  assert.match(calls[0].body.text, /blue cable/);
  assert.deepEqual(played, ['blob:test']);
  assert.match(el('status').textContent, /Speaking with ElevenLabs/);
  el('clear').fire();
  assert.equal(el('speak').disabled, true);
});

test('A failed speech request shows the server message and re-enables Speak', async () => {
  const fetch = async () => ({ok: false, json: async () => ({error: 'ElevenLabs quota or rate limit reached.'})});
  const el = harness(fetch, {atob, Uint8Array, Blob, Audio: class {}, URL});
  el('sample').fire(); el('replay').fire();
  await el('speak').fire();
  assert.match(el('status').textContent, /quota/);
  assert.equal(el('speak').disabled, false);
});

function speechHarness() {
  const calls = [];
  const fetch = async (url, options) => {
    calls.push({url, body: JSON.parse(options.body)});
    return {ok: true, json: async () => ({audio: 'AAAA'})};
  };
  class FakeAudio {
    addEventListener() {}
    play() { return Promise.resolve(); }
    pause() {}
  }
  const el = harness(fetch, {
    atob: (text) => Buffer.from(text, 'base64').toString('binary'),
    Uint8Array, Blob, Audio: FakeAudio,
    URL: {createObjectURL: () => 'blob:test', revokeObjectURL() {}},
  });
  return {el, calls};
}

test('Automatic speaking is opt-in: nothing is sent to ElevenLabs unless it is on', async () => {
  const {el, calls} = speechHarness();
  el('sample').fire();
  el('replay').fire();
  await Promise.resolve();
  assert.equal(calls.length, 0);
  assert.equal(el('speak').disabled, false);
});

test('With automatic speaking on, each new answer is spoken once', async () => {
  const {el, calls} = speechHarness();
  el('auto').checked = true;
  el('sample').fire();
  el('replay').fire();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/speak');
  assert.match(calls[0].body.text, /blue cable/);
});

test('The result badge names the outcome and clears with the result', () => {
  const el = harness();
  el('sample').fire();
  el('replay').fire();
  assert.equal(el('badge').hidden, false);
  assert.equal(el('badge').textContent, 'Scripted demo');
  el('clear').fire();
  assert.equal(el('badge').hidden, true);
});

