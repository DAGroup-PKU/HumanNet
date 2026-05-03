# HumanNet — Open Human-Centric Video Dataset

Public landing page for **HumanNet**, an open human-centric video corpus
(third-person + egocentric, ~967k hours) for training, evaluating, and
advancing embodied-AI models. Co-authored by **PKU DAGroup** and
**SimpleSilicon**. Ships with a small admin backend so the team can edit
content without a redeploy.

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
| Validation | zod | server-side write validation (incl. `VideoRef` discriminated union) |
| Object storage | Aliyun OSS via `ali-oss` | private bucket, signed URL minted per request |
| Public API rate limit | `express-rate-limit` | 60 req/min/IP on `GET /api/config` |
| Static assets | `public/videos/` | served at `/videos/...` (used by `kind:"local"` refs) |

## Scripts

```bash
npm install                       # install web deps
npm install --prefix server       # install api deps
npm --prefix server run seed      # create SQLite + admin user

npm run dev                       # boots vite (5173) + api (5175) together
npm run dev:web                   # vite only
npm run dev:api                   # api only
npm run build                     # tsc -b + vite build → dist/
npm run build:pages               # static GitHub Pages build → dist/
npm run preview                   # preview the static dist
npm run seed                      # alias of `npm --prefix server run seed`
```

Default admin credentials after `npm run seed` are
**`admin / humannet-admin`**. Change them in production via:

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
│       ├── env.ts            # tiny .env loader (incl. OSS_* vars)
│       ├── db.ts             # SQLite + schema (users, site_config)
│       ├── auth.ts           # JWT issue / verify + requireAdmin
│       ├── schema.ts         # zod write validator (VideoRef union, FooterConfig)
│       ├── shared.ts         # constants shared with the frontend
│       ├── default-config.ts # bundled seed payload (mirrors src/lib/config-defaults)
│       ├── oss.ts            # ali-oss client + signOssRef (graceful when creds missing)
│       ├── render.ts         # StoredSiteConfig → PublicSiteConfig (sign + resolve hrefs)
│       ├── seed.ts           # idempotent seed + schema-aware migration
│       └── routes/
│           ├── auth.ts       # POST /api/auth/login + GET /api/auth/me
│           └── config.ts     # public GET /api/config + admin GET/PUT /api/admin/config
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
│       ├── admin-defaults.ts # initial state (StoredSiteConfig shape)
│       ├── api.ts            # fetch helpers + token storage
│       └── editors/
│           ├── Field.tsx
│           ├── VideoRefInput.tsx  # local | external | OSS toggle + dynamic fields
│           ├── HeroEditor.tsx
│           ├── LinksEditor.tsx
│           ├── TeamEditor.tsx
│           ├── GalleryEditor.tsx
│           └── FooterEditor.tsx   # tagline / chips / columns / copyright
├── docs/_stitch/             # original Stitch HTML + screenshots (not built)
├── ProjectStatus.md, Progress.md, Lesson.md
├── CLAUDE.md, AGENTS.md      # AI-context mirrors of .cursor/rules
├── .env.example              # backend env-var documentation
├── package.json, vite.config.ts, tsconfig*.json
└── README.md (this file)
```

## Architecture in one paragraph

The public site is a static SPA. On mount, `<ConfigProvider>` fetches
`GET /api/config` (which is rate-limited at 60 req/min/IP) and merges
the response over a bundled fallback config (`src/lib/config-defaults.ts`).
Components like `<Hero>`, `<Members>`, `<DataGallery>`, and `<Footer>`
consume `useConfig()` so any change on the admin side propagates on the
next page load. The admin SPA (`/admin/*`) talks to the same Express
service with a JWT obtained through `POST /api/auth/login`. Tokens live
in `localStorage` and expire after `JWT_TTL_SECONDS` (default 8h).

### Two configs, one source of truth

The DB stores a `StoredSiteConfig`: video fields are
`{kind:"local"|"external"|"oss", …}` discriminated unions, footer link
hrefs may use `$key` references. `server/src/render.ts` collapses this
into a `SiteConfig` (videos as plain string URLs, hrefs fully resolved)
on every public read. The admin SPA reads / writes the **stored** shape
via `GET / PUT /api/admin/config` (auth required); the public site
reads the **rendered** shape via `GET /api/config`.

### Video security model (six layers)

| # | Layer | Where |
|---|---|---|
| 1 | RAM sub-account, `oss:GetObject` only on the bucket | Aliyun RAM console |
| 2 | Signed URLs with 1h TTL via `ali-oss` `asyncSignatureUrl` | `server/src/oss.ts` |
| 3 | Bucket Referer whitelist | Aliyun OSS console |
| 4 | App-layer rate limit (60 req/min/IP) | `server/src/routes/config.ts` |
| 5 | `controlsList="nodownload"`, `disablePictureInPicture`, etc. | `src/components/VideoFrame.tsx` |
| 6 | (future) CDN URL signing + Referer at the edge | OSS + Aliyun CDN |

Layers 2 / 4 / 5 are implemented in code. 1 / 3 are configuration the
operator does once in the Aliyun console. 6 is optional — add it if
bandwidth becomes a problem.

## Page sections

The page is composed of 8 stacked sections, all responsive:

1. **Hero** — value prop + live third-person capture preview
2. **DatasetProfile** — 7 statistical visualisations rendered from the dataset metadata (scene / object / task taxonomies, motion distributions, quality dashboards)
3. **DataScale** — 6-cell metric mosaic (total hours, TPV hours, FPV hours, scenes, tasks, objects)
4. **PerspectiveExplorer** — `Tabs` between Exocentric and Egocentric, with full-frame video
5. **DataGallery** — filterable grid of 40 OSS-served raw clips (20 TPV + 20 FPV)
6. **Roadmap** — vertical timeline of upcoming milestones (HumanNet Benchmark v1 + Ego-data scaling-law study)
7. **Members** — co-author organisations (PKU DAGroup + SimpleSilicon)
8. **Waitlist** — early-access form linking to a public Tally form

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

### Secrets & seed

- Set `JWT_SECRET` to a long random string (e.g. `openssl rand -hex 64`).
  The dev placeholder is refused at boot when `NODE_ENV=production`.
- Run `ADMIN_PASSWORD=... npm --prefix server run seed` once to
  rotate the admin password (re-running the seed is safe — existing
  config is preserved unless the schema has changed, in which case
  the seed prints the validation issues and runs a one-shot migration
  to bundled defaults).

### OSS hookup

1. Create a private bucket (e.g. `ss-oss-sites`) in the Aliyun OSS console.
2. Add a Referer whitelist on the bucket: your production domain (and
   `localhost` for dev). Disallow empty Referer.
3. Create a RAM sub-account, attach an inline policy granting only
   `oss:GetObject` on `acs:oss:*:*:ss-oss-sites/*`. Generate AK / SK.
4. Set `OSS_REGION`, `OSS_BUCKET`, `OSS_ACCESS_KEY_ID`,
   `OSS_ACCESS_KEY_SECRET` in `.env`. (The server runs without these
   set, but `kind:"oss"` refs degrade to a clearly-broken `data:` URL
   and a warning is logged at boot.)
5. Upload a clip and edit it from `/admin` — switch the video source
   to "OSS" and enter `bucket` + `key`. Each public page-load mints a
   fresh signed URL with `OSS_URL_TTL_SECONDS` lifetime (default 1h).

### Video security note

Real video protection is **a stack, not a single setting**. The site
implements layers 2 / 4 / 5 from the security table above; layers 1
and 3 are configuration the operator does once. Layer 6 (CDN URL
signing) is optional. Don't rely on `controlsList="nodownload"` alone
— it's the weakest layer.

### Serving in prod

Serve the built `dist/` from the same Express instance to avoid CORS
entirely (`app.use(express.static('dist'))` + a SPA fallback for
`/admin/*` is the recommended pattern). Not shipped in dev because
Vite handles serving during development.

### GitHub Pages static mode

GitHub Pages can host the public landing page, but not the Node API,
admin editor, auth, SQLite storage, or `/api/clip/*` OSS proxy. For
Pages, build with the static flag:

```bash
GITHUB_PAGES_BASE=/ego_sites/ npm run build:pages
```

The static build:

- skips the `/api/config` request and uses bundled defaults only
- omits the `/admin/*` route
- copies `media/curation/video` into `dist/media/curation/video`
- prefixes media URLs with Vite's `base`, so project pages work under
  `https://<owner>.github.io/<repo>/`

This repo includes `.github/workflows/deploy-pages.yml`, which builds
and deploys `dist/` automatically from `main`. If the repository is an
owner site named `<owner>.github.io`, the workflow uses `/` as the base;
otherwise it uses `/<repo>/`.

## Working with this repo

Read `.cursor/rules/` and `Lesson.md` before starting any task.
Every task that produces code must end by:

1. Updating `ProjectStatus.md` / `Progress.md` / `Lesson.md`
2. Committing on a feature branch (`<type>/<scope>`)
3. Calling MCP `interactive_feedback`

(See `.cursor/rules/post-resolution-checklist.mdc`.)
