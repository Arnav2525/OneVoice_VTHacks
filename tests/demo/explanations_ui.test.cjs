const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function setup() {
  const elements = new Map();
  function element() {
    return {hidden: false, children: [], textContent: '',
      addEventListener(name, handler) {this[name] = handler;},
      append(...children) {this.children.push(...children);},
      replaceChildren() {this.children = [];}};
  }
  const document = {
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, element());
      return elements.get(id);
    },
    createElement: element,
  };
  const window = {};
  const calls = [];
  const data = {recording: {active: true}, frame: {available: true},
    explanations: {enabled: true, count: 0, items: []}};
  vm.runInNewContext(fs.readFileSync(path.resolve(__dirname,
    '../../demo/web_ui/explanations.js'), 'utf8'), {
      document, window, fetch: async (url, options) => {
        calls.push({url, body: JSON.parse(options.body)});
        return {ok: true, json: async () => data};
      },
    });
  return {el: id => document.getElementById('explanation-' + id), window, data, calls};
}

test('button sends only an explanation action and keeps results hidden during recording', async () => {
  const h = setup();
  h.window.onevoiceExplanations(h.data);
  await h.el('request').click();
  assert.deepEqual(h.calls, [{url: '/api/action', body: {action: 'explain'}}]);
  assert.equal(h.el('results').hidden, true);
});

test('all revealed explanations use safe text rendering and disabled feature hides control', () => {
  const h = setup();
  h.data.recording.active = false;
  h.data.explanations.items = [
    {number: 1, text: '<script>not markup</script>'},
    {number: 2, error: 'API unavailable'},
  ];
  h.window.onevoiceExplanations(h.data);
  assert.equal(h.el('request').disabled, true);
  assert.equal(h.el('list').children.length, 2);
  assert.equal(h.el('list').children[0].children[1].textContent, '<script>not markup</script>');
  h.data.explanations.enabled = false;
  h.window.onevoiceExplanations(h.data);
  assert.equal(h.el('controls').hidden, true);
});
