(() => {
  const APP_URL = 'http://127.0.0.1:8771/';
  const dialog = document.getElementById('connection-dialog');
  const status = document.getElementById('connection-status');
  const help = document.getElementById('connection-help');
  const retry = document.getElementById('connection-retry');
  if (!dialog || !status || !help || !retry) return;
  let attempt = 0;

  async function appIsRunning() {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2500);
    try {
      const response = await fetch(APP_URL + 'api/ping', { cache: 'no-store', signal: controller.signal });
      return response.ok;
    } catch {
      return false;
    } finally {
      clearTimeout(timer);
    }
  }

  function enterTheCamera() {
    const url = APP_URL + '?from=story';
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {
      location.href = url;
      return;
    }
    const iris = document.createElement('div');
    iris.className = 'connect-iris';
    iris.setAttribute('aria-hidden', 'true');
    document.body.append(iris);
    iris.addEventListener('animationend', () => { location.href = url; }, { once: true });
    setTimeout(() => { location.href = url; }, 1600);
  }

  async function connect() {
    const mine = ++attempt;
    help.hidden = true;
    status.textContent = 'Connecting to the live app…';
    const running = await appIsRunning();
    if (mine !== attempt) return;
    if (!running) {
      status.textContent = 'The live app is not running yet.';
      help.hidden = false;
      return;
    }
    status.textContent = 'Connected. Opening the camera view…';
    dialog.close();
    enterTheCamera();
  }

  document.addEventListener('click', (event) => {
    if (event.target.closest('[data-connect]')) connect();
  });
  retry.addEventListener('click', connect);
  dialog.addEventListener('close', () => { attempt += 1; });
})();
