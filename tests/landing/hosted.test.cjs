"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const { pathToFileURL } = require("node:url");

const site = path.resolve(__dirname, "../../one-voice-working-copy");
const read = (name) => fs.readFileSync(path.join(site, name), "utf8");
const modulePath = pathToFileURL(path.join(site, "hosted.js")).href;

function node(attributes) {
  const attrs = new Map(Object.entries(attributes));
  return {
    hidden: false,
    hasAttribute: (name) => attrs.has(name),
    getAttribute: (name) => attrs.get(name) ?? null,
    setAttribute: (name, value) => attrs.set(name, value),
  };
}

function fakeDocument({ links = [], localOnly = [] }) {
  return {
    querySelectorAll: (selector) => (selector === "a[href]" ? links : selector === ".local-only" ? localOnly : []),
  };
}

const APP = "http://127.0.0.1:8771/?from=story";

test("only real hosted domains count as hosted; localhost in any form is local", async () => {
  const { isHostedHost } = await import(`${modulePath}?t=${Math.random()}`);
  for (const local of ["", "localhost", "127.0.0.1", "[::1]"]) assert.equal(isHostedHost(local), false, local);
  for (const hosted of ["www.onevoice.select", "onevoice.select", "onevoice-two.vercel.app", "onevoice.localhost"]) {
    assert.equal(isHostedHost(hosted), true, hosted);
  }
});

test("on a hosted domain the plain app links go to the landing notice and local-only copy is hidden", async () => {
  const { adaptHostedPage } = await import(`${modulePath}?t=${Math.random()}`);
  const plain = node({ href: APP });
  const handled = node({ href: APP, "data-connect": "" });
  const other = node({ href: "about.html" });
  const commands = node({});
  assert.equal(adaptHostedPage(fakeDocument({ links: [plain, handled, other], localOnly: [commands] }), "www.onevoice.select"), true);
  assert.equal(plain.getAttribute("href"), "index.html?demo=laptop");
  assert.equal(handled.getAttribute("href"), APP);
  assert.equal(other.getAttribute("href"), "about.html");
  assert.equal(commands.hidden, true);
});

test("on localhost nothing on the page changes", async () => {
  const { adaptHostedPage } = await import(`${modulePath}?t=${Math.random()}`);
  const plain = node({ href: APP });
  const commands = node({});
  assert.equal(adaptHostedPage(fakeDocument({ links: [plain], localOnly: [commands] }), "127.0.0.1"), false);
  assert.equal(plain.getAttribute("href"), APP);
  assert.equal(commands.hidden, false);
});

test("the landing page never pings the local app from a hosted domain and opens the notice from a linked visit", () => {
  const main = read("main.js");
  assert.match(main, /import \{isHosted\} from '\.\/hosted\.js';/);
  assert.match(main, /function checkApp\(\)\{\r?\n  if\(isHosted\)return Promise\.reject\(/);
  assert.ok(main.indexOf("if(isHosted)return Promise.reject(") < main.indexOf("fetch(`${appUrl}api/ping`"), "hosted check comes before the fetch");
  assert.match(main, /node\.hidden=node\.dataset\.when!==\(isHosted\?'hosted':'local'\)/);
  assert.match(main, /get\('demo'\)==='laptop'/);
  assert.match(main, /history\.replaceState\(null,'',location\.pathname\)/);
});

test("the dialog carries a friendly hosted message and keeps the terminal help for local use only", () => {
  const html = read("index.html");
  assert.match(html, /<span data-when="hosted">Demo runs on<br><em>our laptop\.<\/em><\/span>/);
  assert.match(html, /<p data-when="hosted">The listening app needs a camera, a microphone and a GPU, so it runs on our demo laptop\. Find us at the booth to try it\.<\/p>/);
  assert.match(html, /<p data-when="local">The local app is not responding on port 8771\./);
  assert.match(html, /<button id="retry-connection" class="connect-button" data-when="local">/);
  assert.match(html, /<button id="back-to-story">/);
});

test("About and the demo preview load the hosted helper, and About hides its terminal instructions off localhost", () => {
  for (const page of ["about.html", "demo.html"]) {
    assert.match(read(page), /<script type="module" src="hosted\.js"><\/script><\/body><\/html>/, page);
  }
  assert.match(read("about.html"), /<p class="local-only">The demo runs on this computer\./);
  assert.match(read("package.json"), /node --check hosted\.js/);
});
