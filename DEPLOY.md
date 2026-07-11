# KEPLER — Deployment Guide

Take the local code in this repo to a production stack: **Modal L4 GPU pipeline + Vercel Hono edge + Cloudflare R2 storage + Anthropic Claude verdict streaming.** End-to-end cost is ~$0 for demo scale (Modal's $30 free credit swallows it).

The account signups + OAuth flows must be done by you — I couldn't do them from the coding session. This doc walks each one.

---

## 0. What you need before you start

| Service | Purpose | Sign-up link | Time |
|---|---|---|---|
| **Modal** | GPU pipeline host | https://modal.com | 2 min (OAuth) |
| **Vercel** | Edge functions + web hosting | https://vercel.com | 2 min (OAuth) |
| **Cloudflare** | R2 object storage | https://dash.cloudflare.com | 5 min (bucket + API token) |
| **Anthropic** | Claude Sonnet 4.5 for the verdict | https://console.anthropic.com | 5 min (API key + $5 credit) |

All four have free tiers that cover a hackathon demo comfortably.

---

## 1. Pipeline → Modal

Modal packages the four vision models (CoTracker3 + Depth Anything V2 + stub SAM 2 + stub VGGT) plus the physics fit into one Python app. The container preloads models via `@modal.enter()` so a warm invocation only pays inference cost.

```powershell
cd C:\Users\DzhanybekZakiriiaev\Desktop\kepler\pipeline

# One-time — Modal CLI + browser OAuth.
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv sync
uv run modal setup

# Deploy.
uv run modal deploy modal_app.py
```

Modal prints two URLs at the end:

```
✓ App deployed! 🎉

View Deployment: https://modal.com/apps/<username>/kepler-pipeline
Web endpoint:    https://<username>--kepler-pipeline-web.modal.run
```

**Save the web endpoint URL — it's what the edge layer talks to.**

Verify:

```powershell
curl https://<username>--kepler-pipeline-web.modal.run/health
# {"status":"ok","service":"kepler-pipeline","mode":"modal"}
```

First real `/analyze` request will spend ~1 minute downloading CoTracker3 + Depth Anything V2 weights into the Modal Volumes (`kepler-torch-hub`, `kepler-hf-cache`). Subsequent requests are cached at ~5–10s per clip.

Optional smoke test that runs the full pipeline against a synthesised MP4 without going through HTTP:

```powershell
uv run modal run modal_app.py
```

---

## 2. Cloudflare R2 (video uploads + PLY hosting)

1. **Dashboard → R2 → Create bucket.** Name it `kepler-uploads` (must match `R2_BUCKET` env var). Region: automatic.
2. **Bucket → Settings → CORS policy** → add:

    ```json
    [
      {
        "AllowedOrigins": ["http://localhost:5174", "https://YOUR-WEB-DOMAIN.vercel.app"],
        "AllowedMethods": ["PUT", "GET", "HEAD"],
        "AllowedHeaders": ["*"],
        "MaxAgeSeconds": 3600
      }
    ]
    ```

3. **Right-side panel → Manage R2 API Tokens → Create API Token.**
    - Permissions: **Object Read & Write**
    - Bucket: `kepler-uploads` only
    - Save the Account ID, Access Key ID, Secret Access Key — you'll paste them into Vercel env in a moment.

---

## 3. Anthropic API key

1. https://console.anthropic.com → **API Keys → Create Key**
2. Name: `kepler-verdict`
3. Save the `sk-ant-…` key.
4. Add $5 to your credit balance (once) — a verdict is ~$0.006 so this lasts thousands of runs.

---

## 4. Edge → Vercel

```powershell
cd C:\Users\DzhanybekZakiriiaev\Desktop\kepler\edge

# One-time.
npm install
npm i -g vercel
vercel login
vercel link          # create project "kepler-edge" or attach to an existing one

# Environment variables. Do this for BOTH production AND preview scopes.
vercel env add MODAL_PIPELINE_URL production
vercel env add R2_ACCOUNT_ID          production
vercel env add R2_ACCESS_KEY_ID       production
vercel env add R2_SECRET_ACCESS_KEY   production
vercel env add R2_BUCKET              production   # = kepler-uploads
vercel env add ANTHROPIC_API_KEY      production
vercel env add ALLOWED_ORIGIN         production   # = https://YOUR-WEB-DOMAIN.vercel.app

vercel deploy --prod
```

Vercel prints the deployed URL like `https://kepler-edge-abc123.vercel.app`. Save it — that's `VITE_EDGE_URL` for the web app.

Verify:

```powershell
curl -X POST https://kepler-edge-abc123.vercel.app/api/upload -H "Content-Type: application/json" -d "{}"
# Should return a JSON with a presigned R2 URL + key.
```

---

## 5. Web → Vercel

```powershell
cd C:\Users\DzhanybekZakiriiaev\Desktop\kepler\web

npm install
vercel link          # create project "kepler-web"

# Only one env var needed at the browser level.
vercel env add VITE_EDGE_URL production   # = https://kepler-edge-abc123.vercel.app

vercel deploy --prod
```

Vercel prints the web URL. Open it — drop a video — watch the 3D scene render.

---

## 6. Local development

You don't have to deploy every time. For iterating without touching Modal / Vercel:

**Pipeline (local, stub models — fast):**

```powershell
cd C:\Users\DzhanybekZakiriiaev\Desktop\kepler\pipeline
uv run uvicorn kepler_pipeline.app:app --host 127.0.0.1 --port 8001 --reload
```

**Edge (local, hits the deployed Modal pipeline):**

```powershell
cd C:\Users\DzhanybekZakiriiaev\Desktop\kepler\edge
vercel dev            # picks up .env from `vercel env pull .env.local` if you want
```

**Web (local, points at whatever URL you configure):**

```powershell
cd C:\Users\DzhanybekZakiriiaev\Desktop\kepler\web

# Fastest path: hit the local stub pipeline directly. No edge, no R2, no Claude.
#   → don't set VITE_EDGE_URL; the client falls back to VITE_PIPELINE_URL then localhost.
$env:VITE_PIPELINE_URL = "http://127.0.0.1:8001"
npm run dev
```

If you want to also test the edge layer locally, run `vercel dev` (port 3000), set `VITE_EDGE_URL=http://localhost:3000`, and run the web `npm run dev` on 5174.

---

## 7. End-to-end verification

After all four services are up:

1. Open the deployed web URL.
2. Drop a short (3–6 s) video with visible motion.
3. Web posts to `/api/upload` (edge) → gets a presigned PUT URL for R2.
4. Web PUTs the video directly to R2.
5. Web posts `{ key }` to `/api/analyze` (edge).
6. Edge presigns a GET URL for R2, calls Modal `/analyze` (multipart forwarded).
7. Modal container runs the pipeline, writes PLY + JSON to Modal Volume, returns response.
8. Edge relays the response back to the browser.
9. Browser fetches the PLY from `<modal-url>/artifacts/<id>/point_cloud.ply` and renders in R3F.
10. Browser posts `{ verdict_score, residuals }` to `/api/verdict` (edge) → SSE token stream from Claude.

Success looks like: a rotating colored point cloud, glowing red trajectories, an SVG timeline of σ, and a verdict card that reads *"Consistent with real physics"* for a real video or *"Physically implausible"* for AI-generated content.

---

## 8. Cost per analysis (real numbers)

| Component | Per-analysis cost |
|---|---|
| Modal L4 GPU, ~10 s warm | $0.002 |
| Anthropic Sonnet 4.5 verdict (~500 in, ~300 out tokens) | $0.006 |
| R2 storage + egress | ~$0 |
| Vercel edge functions | $0 (free tier fits 100k requests/mo) |
| **Total** | **~$0.008** |

Modal's $30 free credit covers ~3,750 analyses.

---

## 9. Troubleshooting

- **`modal deploy` fails with "cannot find kepler_physics"** — the physics module is `add_local_dir`-copied from `../physics/kepler_physics`. Make sure that path exists relative to `pipeline/modal_app.py`.
- **First `/analyze` on Modal takes >2 min** — weights are downloading. Watch logs in the Modal dashboard. Subsequent calls hit the cached Volumes.
- **R2 upload fails with CORS error** — the browser is being blocked by R2's default CORS. Add the exact web origin under bucket → Settings → CORS.
- **Vercel edge 504 timeout** — Modal cold start took too long; retry once and it'll be warm. If it's consistent, pin `min_containers=1` on the Modal `@app.cls`.
- **Claude verdict streams then cuts off** — check the Vercel function log for token errors; may be an Anthropic rate limit or missing credit balance.
