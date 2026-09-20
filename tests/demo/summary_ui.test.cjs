const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const htmlPath = path.resolve(__dirname, '../../demo/web_ui/index.html');
const scriptPath = path.resolve(__dirname, '../../demo/web_ui/summary.js');

function harness() {
  const elements = new Map();
  function element() {
    return {value: '', textContent: '', hidden: false, children: [], listeners: {},
      addEventListener(name, fn) { this.listeners[name] = fn; },
      replaceChildren() { this.children = []; }, append(child) { this.children.push(child); },
      setAttribute(name, value) { this[name] = value; },
      fire(name = 'click') { return this.listeners[name](); }};
  }
  const html = fs.readFileSync(htmlPath, 'utf8');
  for (const [, id] of html.matchAll(/id="(summary-[^"]+)"/g)) elements.set(id, element());
  const calls = [];
  const document = {getElementById: id => {assert.ok(elements.has(id), id); return elements.get(id);}, createElement: element};
  const context = {
    document, window: {addEventListener() {}},
    fetch: async (url) => { calls.push(url); throw Error('The summary panel must not make requests'); },
  };
  vm.runInNewContext(fs.readFileSync(scriptPath, 'utf8'), context);
  return {el: name => elements.get('summary-' + name), calls, window: context.window};
}

const points = (h) => JSON.parse(JSON.stringify(h.el('points').children.map((li) => li.textContent)));

test('a summary for a recording appears in the panel and updates the dock label', () => {
  const h = harness();
  h.window.onevoiceRecordingSummary({path: 'a', status: 'loading'});
  assert.equal(h.el('view').textContent, 'Summary preparing…');
  assert.match(h.el('status').textContent, /Gemini is summarizing/);
  h.window.onevoiceRecordingSummary({path: 'a', status: 'ready', result: {summary: 'Recap.', key_points: ['One', 'Two']}});
  assert.equal(h.el('view').textContent, 'View summary');
  assert.equal(h.el('text').textContent, 'Recap.');
  assert.deepEqual(points(h), ['One', 'Two']);
  assert.match(h.el('status').textContent, /saved with your recording/);
});

test('a failed summary shows its error and leaves no stale text behind', () => {
  const h = harness();
  h.window.onevoiceRecordingSummary({path: 'a', status: 'ready', result: {summary: 'Old.', key_points: ['x']}});
  h.window.onevoiceRecordingSummary({path: 'b', status: 'error', error: 'Gemini quota reached.'});
  assert.equal(h.el('view').textContent, 'Summary needs attention');
  assert.equal(h.el('status').textContent, 'Gemini quota reached.');
  assert.equal(h.el('text').textContent, '');
  assert.deepEqual(points(h), []);
});

test('the same recording state is not applied twice', () => {
  const h = harness();
  h.window.onevoiceRecordingSummary({path: 'a', status: 'ready', result: {summary: 'Recap.', key_points: []}});
  h.el('text').textContent = 'edited';
  h.window.onevoiceRecordingSummary({path: 'a', status: 'ready', result: {summary: 'Recap.', key_points: []}});
  assert.equal(h.el('text').textContent, 'edited');
});

test('the View summary button shows and hides the panel', () => {
  const h = harness();
  h.el('panel').hidden = true;
  h.el('view').fire();
  assert.equal(h.el('panel').hidden, false);
  assert.equal(h.el('view')['aria-expanded'], 'true');
  h.el('view').fire();
  assert.equal(h.el('panel').hidden, true);
});

test('there is no manual summarize button, and the panel never calls the server itself', () => {
  const html = fs.readFileSync(htmlPath, 'utf8');
  for (const id of ['summary-run', 'summary-speak', 'summary-auto', 'summary-sample', 'summary-clear', 'summary-badge']) {
    assert.doesNotMatch(html, new RegExp(`id="${id}"`), id);
  }
  const source = fs.readFileSync(scriptPath, 'utf8');
  assert.doesNotMatch(source, /fetch\(|api\/summarize|api\/speak|AbortController/);
  const h = harness();
  h.window.onevoiceRecordingSummary({path: 'a', status: 'ready', result: {summary: 'x', key_points: []}});
  assert.deepEqual(h.calls, []);
});
