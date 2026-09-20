"use strict";

(() => {
  const el = (name) => document.getElementById(`summary-${name}`);
  let recordingSummaryKey = null;

  window.onevoiceRecordingSummary = (item) => {
    if (!item) return;
    const key = `${item.path}|${item.status}`;
    if (key === recordingSummaryKey) return;
    recordingSummaryKey = key;
    el("view").textContent = item.status === "loading"
      ? "Summary preparing…"
      : item.status === "ready" ? "View summary" : "Summary needs attention";
    clearResult();
    if (item.status === "ready") {
      show(item.result);
      el("status").textContent = "Summary saved with your recording.";
    } else {
      el("status").textContent = item.error || "Gemini is summarizing your recording…";
    }
  };

  function clearResult() {
    el("text").textContent = "";
    el("points").replaceChildren();
  }

  function show(result) {
    el("text").textContent = result.summary;
    el("points").replaceChildren();
    result.key_points.forEach((point) => {
      const item = document.createElement("li");
      item.textContent = point;
      el("points").append(item);
    });
  }

  el("view").addEventListener("click", () => {
    const opening = el("panel").hidden;
    el("panel").hidden = !opening;
    el("view").setAttribute("aria-expanded", String(opening));
    if (opening) {
      el("panel").scrollIntoView?.({behavior: "smooth", block: "nearest"});
      el("heading").focus?.({preventScroll: true});
    }
  });
  window.addEventListener("pagehide", clearResult);
})();
