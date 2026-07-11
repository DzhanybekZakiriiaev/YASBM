# KEPLER — edge layer

Vercel-hosted HTTP layer that sits between the browser and the KEPLER
backends. Built with [Hono](https://hono.dev). Three routes:

| Route | Purpose | Runtime |
|---|---|---|
| `POST /api/upload` | Presigned R2 PUT URL for the raw video | Vercel Node |
| `POST /api/analyze` | Kicks off the Modal pipeline, relays SSE | Vercel Node |
| `POST /api/verdict` | Streams Claude Sonnet 4.5 verdict card as SSE | Vercel Node |

The Modal pipeline pulls the uploaded video via a presigned R2 GET URL,
so bytes never flow through the edge.

---

## Local dev

```bash
npm install
cp .env.example .env.local        # fill in real values
npm run dev                       # runs `vercel dev` on :3000
```

`vercel dev` reads `.env.local` automatically. The web app should point
to `http://localhost:3000` for the three `/api/*` routes.

Type check:

```bash
npm run typecheck
```

---

## Environment variables

All required. See `.env.example` for the full list.

| Var | Notes |
|---|---|
| `MODAL_PIPELINE_URL` | Root URL of the Modal FastAPI asgi_app (no trailing slash needed) |
| `R2_ACCOUNT_ID` | Cloudflare account ID (dashboard, top right of R2 page) |
| `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | R2 API token with Object Read & Write |
| `R2_BUCKET` | Bucket name, e.g. `kepler-uploads` |
| `ANTHROPIC_API_KEY` | From console.anthropic.com |
| `ALLOWED_ORIGIN` | Web app origin — e.g. `http://localhost:5174` locally, prod URL in prod |

---

## Deploy

Prereqs: an Anthropic key, an R2 bucket + API token, a deployed Modal
pipeline URL, a Vercel account.

```bash
npm install
vercel login
vercel link                       # attach this directory to a Vercel project

# add every var from .env.example — prompt for value, pick "Production, Preview, Development"
vercel env add MODAL_PIPELINE_URL
vercel env add R2_ACCOUNT_ID
vercel env add R2_ACCESS_KEY_ID
vercel env add R2_SECRET_ACCESS_KEY
vercel env add R2_BUCKET
vercel env add ANTHROPIC_API_KEY
vercel env add ALLOWED_ORIGIN     # set to your production web origin

vercel deploy --prod
```

The routes come up at:

```
https://<your-project>.vercel.app/api/upload
https://<your-project>.vercel.app/api/analyze
https://<your-project>.vercel.app/api/verdict
```

Point the web app's `VITE_EDGE_URL` at the base and go.

---

## R2 bucket CORS

The browser uploads directly to R2 via the presigned PUT URL, so the R2
bucket itself needs a CORS policy that allows PUT from `ALLOWED_ORIGIN`.
Set it once via `wrangler` or the Cloudflare dashboard:

```json
[
  {
    "AllowedOrigins": ["http://localhost:5174", "https://kepler.example.com"],
    "AllowedMethods": ["PUT", "GET"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 3600
  }
]
```

---

## Notes

- `/api/upload` is small and stateless; if you'd rather run it on the
  Vercel Edge runtime, split the R2 client into its own bundle — the
  AWS SDK is heavier than the Edge runtime tolerates.
- `/api/analyze` transparently supports both JSON and SSE from Modal.
  It picks based on the client `Accept` header (`text/event-stream` →
  SSE relay; anything else → JSON pass-through). If Modal doesn't
  expose `/analyze/sse`, hitting the SSE path will 502.
- `/api/verdict` uses `claude-sonnet-4-5`. The system prompt in
  `src/verdict-prompt.ts` enforces the "never claim AI-generated" rule.
