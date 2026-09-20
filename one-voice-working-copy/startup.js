(() => {
  if (location.protocol === 'file:') {
    location.replace('http://127.0.0.1:4319/');
    return;
  }
  import('./main.js?restored=2').catch((error) => console.error('One Voice startup failed', error));
})();
