/* =====================================================================
   PROJECT NEBULA — MOBILE LANDING — interactive layer
   ===================================================================== */
(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* -------------------------------------------------------------------
   * Perspective Explorer — accessible tablist
   * ------------------------------------------------------------------- */
  const tabs = document.querySelectorAll('[role="tab"]');
  const panels = {
    exo: document.getElementById("m-panel-exo"),
    ego: document.getElementById("m-panel-ego"),
  };

  function activateTab(targetId) {
    tabs.forEach((t) => {
      const active = t.dataset.target === targetId;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", active ? "true" : "false");
      t.tabIndex = active ? 0 : -1;
    });

    Object.entries(panels).forEach(([id, panel]) => {
      if (!panel) return;
      const active = id === targetId;
      panel.classList.toggle("is-active", active);
      panel.hidden = !active;
      try {
        if (active) {
          panel.currentTime = 0;
          panel.play().catch(() => {});
        } else {
          panel.pause();
        }
      } catch (_) {}
    });
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => activateTab(tab.dataset.target));
    tab.addEventListener("keydown", (e) => {
      const order = ["exo", "ego"];
      const idx = order.indexOf(tab.dataset.target);
      if (e.key === "ArrowRight" || e.key === "ArrowDown") {
        e.preventDefault();
        const next = order[(idx + 1) % order.length];
        document.querySelector(`[data-target="${next}"]`).focus();
        activateTab(next);
      }
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
        e.preventDefault();
        const prev = order[(idx - 1 + order.length) % order.length];
        document.querySelector(`[data-target="${prev}"]`).focus();
        activateTab(prev);
      }
    });
  });

  /* -------------------------------------------------------------------
   * Rolling frame counter on explorer
   * ------------------------------------------------------------------- */
  const frameTag = document.querySelector("[data-frame-count]");
  if (frameTag && !reduceMotion) {
    let elapsedMs = 14318;
    setInterval(() => {
      elapsedMs += 60;
      const m = String(Math.floor(elapsedMs / 60000)).padStart(2, "0");
      const s = String(Math.floor((elapsedMs % 60000) / 1000)).padStart(2, "0");
      const ms = String(elapsedMs % 1000).padStart(3, "0");
      frameTag.textContent = `${m}:${s}.${ms}`;
    }, 600);
  }

  /* -------------------------------------------------------------------
   * Hero stage play button — toggle controls
   * ------------------------------------------------------------------- */
  const heroVideo = document.querySelector(".hero-stage__media");
  const heroPlay = document.querySelector(".hero-stage__play");
  if (heroPlay && heroVideo) {
    heroPlay.addEventListener("click", () => {
      heroVideo.controls = true;
      heroVideo.muted = false;
      heroVideo.currentTime = 0;
      heroVideo.play().catch(() => {});
      heroPlay.style.display = "none";
    });
  }

  /* -------------------------------------------------------------------
   * Terminal form — simulated CLI feedback
   * ------------------------------------------------------------------- */
  const form = document.querySelector(".terminal__form");
  const feedback = document.querySelector("[data-feedback]");

  if (form && feedback) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const input = form.querySelector("input[type=email]");
      const value = (input?.value || "").trim();
      const valid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);

      feedback.hidden = false;
      if (!valid) {
        feedback.dataset.state = "err";
        feedback.textContent = "✕  invalid_email — handshake aborted.";
        return;
      }

      feedback.dataset.state = "ok";
      feedback.textContent = "✓  uplink_acknowledged — token issued in T-72h.";
      input.value = "";
      input.disabled = true;
    });
  }
})();
