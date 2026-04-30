# Project Nebula — Open-Source Embodied AI Dataset

Public landing page for an open-source embodied-AI dataset & toolchain.
Originally drafted from two separate Stitch designs (web + mobile) and
unified into a single responsive React + HeroUI v3 project.

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Build | Vite 7 | dev / build |
| UI | React 19 + TypeScript 5 | strict typing, .tsx everywhere |
| Styling | Tailwind CSS **v4** | CSS-first config in `src/index.css` |
| Components | HeroUI **v3 (Beta)** | `@heroui/styles` + `@heroui/react` |
| Static assets | `public/videos/` | served at `/videos/...` by Vite |

## Scripts

```bash
npm install             # install deps
npm run dev             # vite dev (default :5174)
npm run build           # tsc -b + vite build → dist/
npm run preview         # preview dist/ on a local server
```

## Layout

```
ego_sites/
├── .cursor/
│   ├── rules/                # source-of-truth Cursor rules
│   └── skills/skill-template # meta-template for new skills
├── public/
│   └── videos/
│       ├── exo/              # third-person clips (~5:3 aspect)
│       └── ego/              # first-person six-panel clips (16:9)
├── src/
│   ├── main.tsx              # React entry
│   ├── App.tsx               # composes the 8 sections
│   ├── index.css             # Tailwind v4 + HeroUI tokens + brand vars
│   ├── components/
│   │   ├── Section.tsx       # eyebrow / headline / body wrapper
│   │   ├── VideoFrame.tsx    # uncropped video w/ aspect-ratio container
│   │   ├── LinkButton.tsx    # branded anchor (link styled as button)
│   │   ├── Navbar.tsx
│   │   ├── Hero.tsx
│   │   ├── ProjectIntro.tsx
│   │   ├── DataScale.tsx
│   │   ├── PerspectiveExplorer.tsx
│   │   ├── DataGallery.tsx
│   │   ├── Roadmap.tsx
│   │   ├── Members.tsx
│   │   ├── Waitlist.tsx
│   │   └── Footer.tsx
│   └── data/                 # static data (members / metrics / clips)
├── docs/_stitch/             # original Stitch HTML + screenshots (not built)
├── ProjectStatus.md, Progress.md, Lesson.md
├── CLAUDE.md, AGENTS.md      # AI-context mirrors of .cursor/rules
├── package.json, vite.config.ts, tsconfig*.json
└── README.md (this file)
```

## Page sections

The page is composed of 8 stacked sections, all responsive:

1. **Hero** — value prop + live exocentric capture preview
2. **ProjectIntro** — four pillars (Open / Sim-to-real / Cross-embodiment / Honest)
3. **DataScale** — 6-cell metric mosaic (hours, clips, tasks, robots, envs, modalities)
4. **PerspectiveExplorer** — `Tabs` between Exocentric and Egocentric, with full-frame video
5. **DataGallery** — filterable grid of raw clips
6. **Roadmap** — vertical timeline of shipped / active / queued milestones
7. **Members** — team / contributor cards
8. **Waitlist** — email capture form (HeroUI `Form` + `TextField` + `Select`)

## Design decisions

- **Videos are never cropped.** The previous version used
  `transform: scale(3)` to crop into the top-left RGB panel of six-panel
  research clips. That hid information that researchers actually want
  to see. The current `<VideoFrame>` uses `object-fit: contain` and an
  `aspect-ratio` container, so the entire frame is always visible.
  Slight letterboxing / pillarboxing is acceptable — the data is.
- **Single responsive project, not separate web / mobile.** All sections
  use mobile-first Tailwind breakpoints (`sm`, `md`, `lg`, `xl`).
- **HeroUI v3 (Beta) for primitives.** Tabs, Card, Form, TextField, Input,
  Select, Chip, Button. Theming is done by overriding HeroUI semantic
  tokens (e.g. `--accent`) inside `.dark { ... }` in `src/index.css`.
- **Anchors look like buttons via `<LinkButton>`.** HeroUI's `Button`
  warns when its `render` prop returns an anchor instead of a button;
  we use a thin custom anchor styled with brand tokens for navigation
  links, and keep `Button` for true form actions.

## Working with this repo

Read `.cursor/rules/` and `Lesson.md` before starting any task.
Every task that produces code must end by:

1. Updating `ProjectStatus.md` / `Progress.md` / `Lesson.md`
2. Committing on a feature branch (`<type>/<scope>`)
3. Calling MCP `interactive_feedback`

(See `.cursor/rules/post-resolution-checklist.mdc`.)
