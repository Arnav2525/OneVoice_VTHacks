(() => {
  const loading = document.getElementById('loading');
  const label = document.getElementById('loading-text');
  const recovery = document.createElement('div');
  recovery.style.cssText = 'max-width:360px;padding:0 24px;text-align:center;line-height:1.7;letter-spacing:.3px';
  recovery.hidden = true;
  const message = document.createElement('p');
  const retry = document.createElement('button');
  retry.textContent = 'RETRY LOADING';
  retry.style.cssText = 'margin-top:18px;padding:12px 20px;border:1px solid #edf4ff88;background:#223148;color:white;font:inherit;letter-spacing:1px';
  retry.addEventListener('click', () => location.reload());
  recovery.append(message, retry);
  loading.append(recovery);
  function explain(text) {
    if (loading.classList.contains('done')) return;
    label.textContent = 'THE SCENE IS TAKING LONGER THAN EXPECTED';
    message.textContent = text;
    recovery.hidden = false;
  }
  const timer = setTimeout(() => explain('You can keep waiting, or retry the local preview.'), 20000);
  const observer = new MutationObserver(() => {
    if (loading.classList.contains('done')) { clearTimeout(timer); observer.disconnect(); }
  });
  observer.observe(loading, { attributes: true, attributeFilter: ['class'] });
  if (location.protocol === 'file:') {
    clearTimeout(timer);
    label.textContent = 'OPEN THE LOCAL PREVIEW';
    message.textContent = 'This 3D page needs its local server. Run Start One Voice.cmd, then open the preview below.';
    retry.hidden = true;
    const link = document.createElement('a');
    link.href = 'http://127.0.0.1:4319/';
    link.textContent = 'OPEN ONE VOICE ↗';
    link.style.cssText = 'display:block;margin-top:18px;text-decoration:underline';
    recovery.append(link); recovery.hidden = false;
    return;
  }
  import('./main.js?restored=2').catch(error => {
    clearTimeout(timer);
    console.error('One Voice startup failed', error);
    explain('A required page script could not load. Retry to fetch the restored version.');
  });
})();
