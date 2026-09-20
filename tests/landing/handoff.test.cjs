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

test("the launcher runs both servers with no console windows, waits for them, then opens the story page", () => {
  const ps = read(repo, "scripts", "start_onevoice.ps1");
  const cmd = read(repo, "Start OneVoice Experience.cmd");
  const vbs = read(repo, "Start OneVoice Experience.vbs");
  const stop = read(repo, "scripts", "stop_onevoice.ps1");
  assert.match(ps, /\$style = if \(\$Visible\) \{ 'Normal' \} else \{ 'Hidden' \}/);
  assert.equal((ps.match(/-WindowStyle \$style/g) || []).length, 2);
  assert.match(ps, /'--port', '8771'/);
  assert.match(ps, /server\.cjs/);
  assert.match(ps, /RedirectStandardOutput/);
  assert.match(ps, /ValidateSet\('preview', 'live'\)\]\[string\]\$Mode = 'preview'/);
  assert.ok(ps.indexOf("Start-Process $storyUrl") > ps.indexOf("$deadline"), "browser opens only after the wait");
  assert.doesNotMatch(cmd, /start "OneVoice|cmd \/k/);
  assert.match(cmd, /set "MODE=preview"/);
  assert.match(cmd, /"%%~A"=="live"/);
  assert.match(cmd, /start_onevoice\.ps1/);
  assert.match(vbs, /, 0, False/);
  assert.match(stop, /8771/);
  assert.match(stop, /4319/);
});

test("the SAY HELLO button is nudged right of the yeti and kept on screen so the case never covers it", () => {
  const scene = read(site, "scene.js");
  assert.match(scene, /helloWidth\*\.4/);
  assert.match(scene, /width-helloWidth\*\.5-16/);
});

test("About packs the glasses into the case, then opens About with the yeti arriving", () => {
  const scene = read(site, "scene.js");
  const about = read(site, "about.html");
  const yeti = read(site, "about-yeti.js");
  assert.match(html, /id="travel-toggle"[^>]*>ABOUT \/ PACK UP/);
  assert.match(html, /<nav aria-label="Main navigation"><a href="#top" class="home-link">Home<\/a>/);
  assert.doesNotMatch(html, /data-about/);
  assert.match(main, /function openAbout\(\)/);
  assert.match(main, /world\.travel\(\)/);
  assert.match(main, /about\.html\?arrival=yeti/);
  assert.match(main, /is-travelling/);
  assert.match(scene, /import \{makeSuitcase\} from '\.\/journey\.js'/);
  assert.match(scene, /createWorld\(canvas, onReady, onPacked\)/);
  assert.match(scene, /suitcase\.hinge\.rotation\.x/);
  assert.match(scene, /onPacked\(\)/);
  assert.match(about, /has\("arrival"\)/);
  assert.match(yeti, /about-landed/);
});

test("the case animation fades the landing UI and the About arrival has its styles", () => {
  const css = read(site, "style.css");
  assert.match(css, /\.is-travelling \.hero-title/);
  assert.match(css, /\.about-arriving \.about-yeti\{transform:translate\(62vw/);
  assert.match(css, /\.about-arriving\.about-landed \.about-yeti/);
  assert.match(css, /@view-transition\{navigation:auto\}/);
});

test("no intermediate screen sits between the lens zoom and the app, and the forest is gone", () => {
  const css = read(site, "style.css");
  const scene = read(site, "scene.js");
  assert.doesNotMatch(main, /Ready when you are|lens-destination/);
  assert.doesNotMatch(css, /lens-destination/);
  assert.doesNotMatch(scene, /forest/i);
  assert.match(css, /\.world #conversation \.cafe\{display:none\}/);
});

test("the yeti stays put while the glasses pack: no fly-in and no growing before About opens", () => {
  const scene = read(site, "scene.js");
  assert.doesNotMatch(scene, /yeti\.root\.scale\.setScalar\(1\.25\+approach/);
  assert.doesNotMatch(scene, /travelTime>=0\)\{const approach/);
  assert.doesNotMatch(scene, /YETI IS SHOWING YOU/);
  assert.match(scene, /yeti\.root\.scale\.setScalar\(1\.25\)/);
  assert.match(scene, /suitcase\.hinge\.rotation\.x/);
});

test("the loading and retry screen is gone and nothing in the scripts still depends on it", () => {
  const scene = read(site, "scene.js");
  const startup = read(site, "startup.js");
  const css = read(site, "style.css");
  assert.doesNotMatch(html, /id="loading"|loading-text/);
  assert.doesNotMatch(main, /#loading/);
  assert.doesNotMatch(scene, /#loading/);
  assert.doesNotMatch(css, /\.loading/);
  assert.doesNotMatch(startup, /RETRY LOADING|TAKING LONGER|loading/);
  assert.match(startup, /import\('\.\/main\.js/);
  assert.match(main, /world-error/);
});
