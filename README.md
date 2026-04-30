# Project Nebula — Open-Source Embodied AI Dataset

Public landing page for an open-source embodied-AI dataset & toolchain,
plus a small admin backend so the team can edit content without a
redeploy.

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Build (web) | Vite 7 | dev / build |
| UI | React 19 + TypeScript 5 | strict typing, .tsx everywhere |
| Routing | react-router-dom 6 | `/` public, `/admin/*` editor |
| Styling | Tailwind CSS **v4** | CSS-first config in `src/index.css` |
| Components | HeroUI **v3 (Beta)** | `@heroui/styles` + `@heroui/react` |
| Backend | Node 22 + Express 4 | `server/`, ESM, `tsx watch` |
| Auth | JWT (HS256) | rate-limited login, 8-hour TTL |
| Storage | SQLite (better-sqlite3) | single `site_config` JSON row |
| Validation | zod | server-side write validation |
| Static assets | `public/videos/` | served at `/videos/...` (dev fallback) |

## Scripts

```bash
npm install                       # install web deps
npm install --prefix server       # install api deps
npm --prefix server run seed      # create SQLite + admin user

npm run dev                       # boots vite (5174) + api (5175) together
npm run dev:web                   # vite only
npm run dev:api                   # api only
npm run build                     # tsc -b + vite build → dist/
npm run preview                   # preview the static dist
npm run seed                      # alias of `npm --prefix server run seed`
```

Default admin credentials after `npm run seed` are
**`admin / nebula-admin`**. Change them in production via:

```bash
JWT_SECRET=<long-random-string> ADMIN_PASSWORD=<new-password> \
  npm --prefix server run seed
```

See `.env.example` for every backend env var.

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
├── server/                   # backend (admin API + content store)
│   ├── package.json
│   ├── tsconfig.json
│   ├── data/                 # SQLite (gitignored)
│   └── src/
│       ├── index.ts          # Express bootstrap
│       ├── env.ts            # tiny .env loader
│       ├── db.ts             # SQLite + schema (users, site_config)
│       ├── auth.ts           # JWT issue / verify + requireAdmin
│       ├── schema.ts         # zod write validator
│       ├── shared.ts         # constants shared with the frontend
│       ├── default-config.ts # bundled seed payload (mirrors src/lib/config-defaults)
│       ├── seed.ts           # idempotent seed CLI
│       └── routes/
│           ├── auth.ts       # POST /api/auth/login + GET /api/auth/me
│           └── config.ts     # GET / PUT /api/config
├── src/
│   ├── main.tsx              # React entry — mounts BrowserRouter + ConfigProvider
│   ├── App.tsx               # public landing (8 sections)
│   ├── index.css             # Tailwind v4 + HeroUI tokens + brand vars
│   ├── components/           # Hero / Navbar / Footer / DataGallery / …
│   ├── data/                 # base data shapes (Member, GalleryClip)
│   ├── lib/
│   │   ├── config-types.ts   # SiteConfig shape (frontend ↔ backend)
│   │   ├── config-defaults.ts# bundled defaults (fallback if API down)
│   │   ├── useConfig.tsx     # React Context + provider hook
│   │   └── links.ts          # default external URLs
│   └── admin/                # admin SPA at /admin/*
│       ├── AdminApp.tsx
│       ├── AdminLogin.tsx
│       ├── AdminDashboard.tsx
│       ├── api.ts            # fetch helpers + token storage
│       └── editors/
│           ├── Field.tsx
│           ├── HeroEditor.tsx
│           ├── LinksEditor.tsx
│           ├── TeamEditor.tsx
│           └── GalleryEditor.tsx
├── docs/_stitch/             # original Stitch HTML + screenshots (not built)
├── ProjectStatus.md, Progress.md, Lesson.md
├── CLAUDE.md, AGENTS.md      # AI-context mirrors of .cursor/rules
├── .env.example              # backend env-var documentation
├── package.json, vite.config.ts, tsconfig*.json
└── README.md (this file)
```

## Architecture in one paragraph

The public site is a static SPA. On mount, `<ConfigProvider>` fetches
`GET /api/config` and merges the response over a bundled fallback
config (`src/lib/config-defaults.ts`). Components like `<Hero>`,
`<Members>`, and `<DataGallery>` consume `useConfig()` so any change
on the admin side propagates on the next page load. The admin SPA
(`/admin/*`) talks to the same Express service with a JWT obtained
through `POST /api/auth/login`. Tokens live in `localStorage` and
expire after `JWT_TTL_SECONDS` (default 8h).

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

## Production deploy notes

- Set `JWT_SECRET` to a long random string. The dev placeholder is
  refused at boot when `NODE_ENV=production`.
- Run `ADMIN_PASSWORD=... npm --prefix server run seed` once to
  rotate the admin password (re-running the seed is safe; existing
  config is left intact).
- Real video protection is **out of the frontend's scope**. The
  `controlsList="nodownload"` etc. on `<video>` are speed bumps, not
  a security boundary. Configure your OSS bucket with a Referer
  whitelist + signed URLs (5-minute STS) for actual access control.
- Serve the built `dist/` from the same Express instance in
  production to avoid CORS entirely (`app.use(express.static('dist'))`
  + a SPA fallback for `/admin/*` is the recommended pattern; not
  shipped in dev because Vite handles it during development).

## Working with this repo

Read `.cursor/rules/` and `Lesson.md` before starting any task.
Every task that produces code must end by:

1. Updating `ProjectStatus.md` / `Progress.md` / `Lesson.md`
2. Committing on a feature branch (`<type>/<scope>`)
3. Calling MCP `interactive_feedback`

(See `.cursor/rules/post-resolution-checklist.mdc`.)
