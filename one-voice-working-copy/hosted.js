const LOCAL_HOSTS = new Set(['', 'localhost', '127.0.0.1', '[::1]']);
const APP_LINK = '127.0.0.1:8771';

export const isHostedHost = (hostname) => !LOCAL_HOSTS.has(hostname);

export function adaptHostedPage(doc, hostname) {
  if (!isHostedHost(hostname)) return false;
  for (const link of doc.querySelectorAll('a[href]')) {
    if (link.hasAttribute('data-connect') || !link.getAttribute('href').includes(APP_LINK)) continue;
    link.setAttribute('href', 'index.html?demo=laptop');
  }
  for (const node of doc.querySelectorAll('.local-only')) node.hidden = true;
  return true;
}

export const isHosted = typeof location !== 'undefined' && isHostedHost(location.hostname);

if (typeof document !== 'undefined' && typeof location !== 'undefined') adaptHostedPage(document, location.hostname);
