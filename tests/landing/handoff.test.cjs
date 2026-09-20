"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const repo = path.resolve(__dirname, "../..");
const site = path.join(repo, "one-voice-working-copy");
const read = (...parts) => fs.readFileSync(path.join(...parts), "utf8");
const html = read(site, "index.html");
const main = read(site, "main.js");
const server = read(site, "server.cjs");
const pkg = JSON.parse(read(site, "package.json"));
const session = read(repo, "demo", "web_session.py");

test("the landing page has the dialog controls the handoff code binds to", () => {
  for (const id of ["connection-dialog", "close-connection", "back-to-story", "retry-connection"]) {
    assert.match(html, new RegExp(`id="${id}"`), id);
  }
  assert.match(html, /data-connect/);
});

test("every TRY DEMO entry points at the local listening app with the story flag", () => {
  const links = [...html.matchAll(/href="(http:\/\/127\.0\.0\.1:8771\/[^"]*)"/g)].map((m) => m[1]);
  assert.ok(links.length >= 1);
  for (const link of links) assert.match(link, /\?from=story$/);
});

test("the handoff pings the app first and only continues when it identifies itself", () => {
  assert.match(main, /const appUrl='http:\/\/127\.0\.0\.1:8771\/'/);
  assert.match(main, /api\/ping/);
  assert.match(main, /\.app!=='onevoice'/);
  assert.match(main, /location\.assign\(`\$\{appUrl\}\?from=story`\)/);
  assert.match(main, /handoff-failed/);
});

test("the story page is served from the port the app's ping endpoint allows", () => {
  assert.match(server, /ONEVOICE_PORT \|\| 4319/);
  assert.match(session, /STORY_ORIGIN = "http:\/\/127\.0\.0\.1:4319"/);
});

test("the older iris connect script is gone so two handlers never bind the same buttons", () => {
  assert.doesNotMatch(html, /connect\.js|connect\.css/);
  assert.equal(fs.existsSync(path.join(site, "connect.js")), false);
  assert.equal(fs.existsSync(path.join(site, "connect.css")), false);
});

test("the landing syntax check covers the new About yeti script", () => {
  assert.match(pkg.scripts.check, /node --check about-yeti\.js/);
  assert.doesNotMatch(pkg.scripts.check, /connect\.js/);
  assert.ok(fs.existsSync(path.join(site, "about-yeti.js")));
});

test("the one-click launcher starts the story page and the app, preview by default", () => {
  const launcher = read(repo, "Start OneVoice Experience.cmd");
  assert.match(launcher, /--port 8771/);
  assert.match(launcher, /server\.cjs/);
  assert.match(launcher, /ONEVOICE_MODE=--preview/);
  assert.match(launcher, /if \/I "%~1"=="live"/);
});
