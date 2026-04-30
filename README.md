# Project Nebula — Embodied AI Dataset Landing

Two faithful, production-grade implementations of the **Project Nebula** landing page,
sourced from Stitch designs:

| Surface  | Stitch project                | Target viewport | Source                                     |
| -------- | ----------------------------- | --------------- | ------------------------------------------ |
| `web/`   | `17037056815451835087`        | 1280px+         | [`web/_stitch/source.html`](web/_stitch)   |
| `mobile/`| `4865892911741871417`         | 360–430px       | [`mobile/_stitch/source.html`](mobile/_stitch) |

Both sites share an **Industrial Hard-Science** design system: matte charcoal surfaces,
razor-sharp 0–8 px corners, `Space Grotesk` headlines, `Inter` body copy,
`JetBrains Mono` numeric readouts, and `#EE9F32` industrial-amber accent.

## Repository layout

```
ego_sites/
├── README.md                  ← you are here
├── ProjectStatus.md           ← work log
├── Progress.md                ← task notes
├── Lesson.md                  ← lessons learned
├── assets/
│   └── videos/
│       ├── exo/               ← exocentric (3rd-person) clips
│       └── ego/               ← egocentric (1st-person) clips
├── web/
│   ├── index.html             ← desktop landing (1280px target)
│   ├── assets/
│   │   ├── styles.css
│   │   └── script.js
│   └── _stitch/               ← original Stitch HTML + screenshot (reference only)
└── mobile/
    ├── index.html             ← mobile landing (390px target)
    ├── assets/
    │   ├── styles.css
    │   └── script.js
    └── _stitch/               ← original Stitch HTML + screenshot (reference only)
```

## Run locally

Both sites are pure static HTML/CSS/JS — no build step.

```bash
# from the repo root
python3 -m http.server 4173
```

Then open:

- Desktop: <http://127.0.0.1:4173/web/index.html>
- Mobile : <http://127.0.0.1:4173/mobile/index.html>

A static server is required (not `file://`) so the relative video paths
(`../assets/videos/...`) resolve.

## What is implemented

Each surface ships a fully interactive landing experience.

### `web/`  — full landing

1. **Top nav** with sticky uplink-signal bar (live coordinates).
2. **Hero** with side-by-side copy + autoplaying telemetry video,
   live HUD readouts (`FRAME`, `DELTA`, `JOINT Σ`) that tick every 600 ms,
   pulsing `LIVE TELEMETRY` chip, and primary/secondary CTAs.
3. **Perspective Explorer** with accessible `role="tab"` tabs for
   Exocentric / Egocentric perspectives, animated cross-fade between
   videos, animated CAM_FRAME counter, and a 6-row camera-parameter sheet.
4. **Deployment Roadmap** — 4-stage horizontal timeline with active /
   done / pending node states.
5. **Raw Data Showcases** — three video cards that play on hover or click.
6. **Developer Beta terminal** — fake CLI mock with spinning
   `awaiting_handshake` indicator and a working email form
   (regex-validated, simulated handshake feedback).
7. Sticky footer with monospace coordinates.

### `mobile/` — focused mobile landing

1. **Sticky app bar** (safe-area aware) with `Access Terminal` action.
2. **Hero** with stacked copy, primary `Get Started`, secondary `Live Demo`,
   live-feed video preview with click-to-unmute play button, and
   crosshair markers. Three metric cards stack below.
3. **Perspective Explorer** with pill tabs (radius-`full`, the only
   exception to the design system's sharp shape language — used here
   purely as a touch-friendly affordance), animated CAM_FRAME counter,
   and a 4-row parameter list.
4. **Roadmap** — vertical timeline.
5. **Developer Beta terminal** with the same form behaviour as web.
6. Footer.

## Interactions

- ✅ Accessible `role="tablist"` / `role="tab"` with arrow-key navigation
  and `aria-selected` state.
- ✅ Hover-to-play / click-to-play on showcase videos.
- ✅ Email form with regex validation; simulated CLI success/error
  feedback rendered inline.
- ✅ Section reveal on scroll (intersection-observer driven).
- ✅ All animations honour `prefers-reduced-motion: reduce`.
- ✅ Telemetry HUD counters update with realistic jitter
  (10–14 ms `DELTA`, 5–7 `JOINT Σ`).

## Design system tokens

Colour ramps, spacing, typography, and component primitives live in
each `assets/styles.css` under `:root`. The systems are intentionally
duplicated rather than shared — per the user instruction the two
designs are independent deliverables targeting different devices, and
each may evolve separately.

| Token            | Value                            |
| ---------------- | -------------------------------- |
| `--base`         | `#111113` (matte charcoal)       |
| `--bg`           | `#131315` (page surface)         |
| `--primary`      | `#EE9F32` (industrial amber)     |
| `--on-primary`   | `#1a0f00`                        |
| `--outline`      | `#2C2C2E` (1px dividers)         |
| `--outline-strong` | `#3a3a3c`                      |
| `--respiration-gap` | `120px`                       |

Typography scale — `display-lg`, `headline-md`, `body-base`, `body-sm`,
`label-mono` — matches the Stitch design tokens exactly.

## Video assets

The hero, explorer, and showcase blocks reference real research clips
under `assets/videos/`:

- `exo/` — third-person reference camera captures (motion-capture
  overlays + meshes baked into a 3-panel layout).
- `ego/` — first-person POV captures (six-panel format, top-left panel
  is the raw RGB stream).

The CSS crops to the top-left panel via
`transform: scale(3.05); transform-origin: 0 0;` plus an industrial
colour grade `filter: contrast(1.18) saturate(0.55) brightness(0.78)`.

## Browser support

Modern evergreen browsers. Uses:

- CSS custom properties, `aspect-ratio`, `clamp()`, `color-mix()`,
  `:has()`-free fallbacks, `backdrop-filter`.
- ES2020 vanilla JS (no framework, no bundler).
- `IntersectionObserver` (with graceful no-op on older browsers).
