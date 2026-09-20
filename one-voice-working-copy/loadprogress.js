const CELLS = 26;
const LIT_GLYPHS = ['=', '+', '=', '=', '+'];

let shown = 0;
let frame = 0;
let timer = null;

const clamp = (value) => Math.min(1, Math.max(0, Number(value) || 0));
const reduced = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

export function asciiBar(fraction, step = 0) {
  const lit = Math.round(clamp(fraction) * CELLS);
  let text = '';
  for (let i = 0; i < CELLS; i += 1) {
    text += i < lit ? LIT_GLYPHS[(i * 7 + step * 3 + (i % 3) * step) % LIT_GLYPHS.length] : '-';
  }
  return { text, lit };
}

function paint() {
  const litNode = document.getElementById('loading-lit');
  const dimNode = document.getElementById('loading-dim');
  if (!litNode || !dimNode) return;
  const { text, lit } = asciiBar(shown, reduced() ? 0 : frame);
  litNode.textContent = text.slice(0, lit);
  dimNode.textContent = text.slice(lit);
  document.getElementById('loading')?.setAttribute('aria-valuenow', String(Math.round(shown * 100)));
}

export function setLoad(fraction) {
  shown = Math.max(shown, clamp(fraction));
  paint();
  return shown;
}

export function resetLoad() {
  shown = 0;
  frame = 0;
}

export function stopLoadingAnimation() {
  if (timer) clearInterval(timer);
  timer = null;
}

export function startLoadingAnimation() {
  if (timer || reduced()) return;
  timer = setInterval(() => {
    frame += 1;
    paint();
    if (document.getElementById('loading')?.classList.contains('done')) stopLoadingAnimation();
  }, 90);
}

export async function readWithProgress(response, from, to) {
  const total = Number(response.headers.get('X-Uncompressed-Length') || response.headers.get('Content-Length')) || 0;
  if (!response.body || !total) {
    const buffer = await response.arrayBuffer();
    setLoad(to);
    return buffer;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    setLoad(from + (to - from) * Math.min(0.98, received / total));
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  setLoad(to);
  return bytes.buffer;
}

export function trackLoadingManager(manager, from, to) {
  manager.onProgress = (_url, loaded, total) => setLoad(from + (to - from) * (total ? loaded / total : 0));
}

if (typeof window !== 'undefined' && document.getElementById('loading')) startLoadingAnimation();
