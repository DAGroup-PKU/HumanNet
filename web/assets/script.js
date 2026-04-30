/* =====================================================================
   PROJECT NEBULA — DESKTOP LANDING — interactive layer
   ===================================================================== */
(() => {
  "use strict";

  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* -------------------------------------------------------------------
   * Perspective Explorer — accessible tablist
   * ------------------------------------------------------------------- */
  const tabs = document.querySelectorAll('[role="tab"]');
  const panels = {
    exo: document.getElementById("panel-exo"),
    ego: document.getElementById("panel-ego"),
  };

  function activateTab(targetId) {
    tabs.forEach((t) => {
      const isActive = t.dataset.target === targetId;
      t.classList.toggle("is-active", isActive);
      t.setAttribute("aria-selected", isActive ? "true" : "false");
      t.tabIndex = isActive ? 0 : -1;
    });

    Object.entries(panels).forEach(([id, panel]) => {
      if (!panel) return;
      const isActive = id === targetId;
      panel.classList.toggle("is-active", isActive);
      panel.hidden = !isActive;
      try {
        if (isActive) {
          panel.currentTime = 0;
          panel.play().catch(() => {});
        } else {
          panel.pause();
        }
      } catch (_) {
        /* video may not be ready yet — ignore */
      }
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
   * Showcase cards — click-to-play, autoplay-on-hover for taste
   * ------------------------------------------------------------------- */
  document.querySelectorAll(".card-clip").forEach((card) => {
    const video = card.querySelector("video");
    const playBtn = card.querySelector(".card-clip__play");
    if (!video) return;

    const tryPlay = () => {
      video.play().catch(() => {});
      card.classList.add("is-playing");
    };
    const pause = () => {
      video.pause();
    };

    if (playBtn) {
      playBtn.addEventListener("click", (e) => {
        e.preventDefault();
        tryPlay();
      });
    }

    if (!reduceMotion) {
      card.addEventListener("mouseenter", tryPlay);
      card.addEventListener("mouseleave", () => {
        if (!card.classList.contains("is-playing-locked")) {
          pause();
          card.classList.remove("is-playing");
        }
      });
    }
  });

  /* -------------------------------------------------------------------
   * Telemetry HUD — animated readouts (frame counter, delta, joints)
   * ------------------------------------------------------------------- */
  const counters = {
    frame: document.querySelector('[data-counter="frame"]'),
    delta: document.querySelector('[data-counter="delta"]'),
    joint: document.querySelector('[data-counter="joint"]'),
  };
  const frameTag = document.querySelector("[data-frame-count]");

  if (!reduceMotion) {
    let frame = 128;
    let elapsedMs = 14318;

    const tick = () => {
      frame += 1;
      elapsedMs += 60;
      if (counters.frame) {
        counters.frame.textContent = String(frame).padStart(6, "0");
      }
      if (counters.delta) {
        const d = (10 + Math.random() * 4).toFixed(2);
        counters.delta.textContent = `${d}ms`;
      }
      if (counters.joint) {
        // joint count flickers slightly (5..7)
        counters.joint.textContent = String(5 + Math.floor(Math.random() * 3));
      }
      if (frameTag) {
        const total = elapsedMs;
        const m = String(Math.floor(total / 60000)).padStart(2, "0");
        const s = String(Math.floor((total % 60000) / 1000)).padStart(2, "0");
        const ms = String(total % 1000).padStart(3, "0");
        frameTag.textContent = `${m}:${s}.${ms}`;
      }
    };
    setInterval(tick, 600);
  }

  /* -------------------------------------------------------------------
   * Terminal handshake spinner
   * ------------------------------------------------------------------- */
  const spinner = document.querySelector("[data-spinner]");
  if (spinner && !reduceMotion) {
    const frames = ["|", "/", "—", "\\"];
    let i = 0;
    setInterval(() => {
      spinner.textContent = frames[i = (i + 1) % frames.length];
    }, 120);
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
        feedback.textContent = "✕  invalid_email — execution halted. handshake aborted.";
        return;
      }

      feedback.dataset.state = "ok";
      feedback.textContent = "✓  uplink_acknowledged — beta_token will be issued to " + value + " in T-72h.";
      input.value = "";
      input.disabled = true;
    });
  }

  /* -------------------------------------------------------------------
   * Soft enter animation for sections (intersection observer)
   * ------------------------------------------------------------------- */
  if (!reduceMotion && "IntersectionObserver" in window) {
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-revealed");
            io.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "-10% 0px -10% 0px" }
    );
    document.querySelectorAll(".section").forEach((s) => io.observe(s));
  }
})();
