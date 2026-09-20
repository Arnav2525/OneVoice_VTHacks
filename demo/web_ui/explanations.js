"use strict";

(() => {
  const button = document.getElementById("explanation-request");
  const status = document.getElementById("explanation-status");
  const results = document.getElementById("explanation-results");
  const list = document.getElementById("explanation-list");
  let pending = false;
  let available = false;
  let lastResults = "";
  let error = "";

  window.onevoiceExplanations = (data) => {
    const explanations = data.explanations;
    document.getElementById("explanation-controls").hidden = !explanations?.enabled;
    available = Boolean(explanations?.enabled && data.recording?.active && data.frame?.available);
    button.disabled = pending || !available;
    status.textContent = error || explanations?.error || (pending ? "Capturing this moment…" :
      explanations?.count ? `${explanations.count} request(s) captured. Explanations appear with your summary after recording.` :
        "");
    const items = explanations?.items || [];
    const key = JSON.stringify(items);
    if (key === lastResults) return;
    lastResults = key;
    results.hidden = items.length === 0;
    list.replaceChildren();
    for (const item of items) {
      const card = document.createElement("article");
      const heading = document.createElement("h4");
      heading.textContent = `Request ${item.number}`;
      const text = document.createElement("p");
      text.textContent = item.text || item.error || "Preparing explanation…";
      card.append(heading, text);
      list.append(card);
    }
  };

  button.addEventListener("click", async () => {
    if (pending || !available) return;
    pending = true;
    error = "";
    button.disabled = true;
    try {
      const response = await fetch("/api/action", {
        method: "POST",
        headers: {"Content-Type": "application/json", "X-OneVoice-UI": "1"},
        body: JSON.stringify({action: "explain"}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not capture this request.");
      window.onevoiceExplanations(data);
    } catch (failure) {
      error = failure.message;
      status.textContent = error;
    } finally {
      pending = false;
      button.disabled = !available;
    }
  });
})();
