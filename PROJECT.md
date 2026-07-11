# KEPLER — Project Brief

**Every generated video breaks a law of physics. We prove which one, and where.**

KEPLER is a browser-based forensic tool that takes a short video clip, reconstructs the scene in 3D, tracks rigid objects, fits a Newtonian trajectory to each, and measures where observed motion diverges from what physics would produce. Real footage passes cleanly. AI-generated footage — Sora 2, Veo 3.1, Kling 3.0 — leaves a physics residual you can visualize, quantify, and point at.

The output is a rotating 3D reconstruction of the scene with two trajectories overlaid: the object as it moved in the video (red) and the object as it *should* have moved given identical release conditions (green). Where they diverge is where the model lied.

---

## Architecture (target)

```
┌─── BROWSER ────────────────────────────────────────────┐
│  Vite + React 19 + TS + Tailwind + R3F + drei          │
│  ├─ Upload / video scrubber                             │
│  ├─ R3F + Three.js viewer  (point cloud + trajectories) │
│  │    with @react-three/postprocessing (bloom + SSAO)   │
│  ├─ Residual timeline chart                             │
│  └─ Verdict card (streamed from Claude Sonnet 4.5)      │
└──────────────┬──────────────────────────────────────────┘
               │ HTTPS + SSE for progress
┌─── EDGE (Vercel + Hono) ──────────────────────────────┐
│  /api/upload   → R2 presigned URL                      │
│  /api/analyze  → Modal function invocation, SSE relay  │
│  /api/verdict  → Claude Sonnet 4.5 stream              │
└──────────────┬────────────────────────────────────────┘
               │
┌─── GPU (Modal, L4 24GB, @app.asgi_app FastAPI) ───────┐
│  Stage 1  SAM 2.1              masks per object        │
│  Stage 2  CoTracker3           dense 2D point tracks   │
│  Stage 3  Video Depth Anything temporally-stable depth │
│  Stage 4  VGGT                 camera pose + cloud     │
│  Stage 5  Lift 2D→3D           world-frame trajectories│
│  Stage 6  Newton fit           kepler_physics.fit      │
│  Stage 7  Package              PLY + tracks + residuals│
└──────────────┬────────────────────────────────────────┘
               │
┌─── STORAGE ───────────────────────────────────────────┐
│  Cloudflare R2       videos + PLY point clouds         │
└───────────────────────────────────────────────────────┘
```

---

## Directory layout

```
kepler/
├── PROJECT.md              this file
├── README.md               run + deploy instructions
├── DEPLOY.md               step-by-step deployment (Modal, Vercel, R2, Anthropic)
├── OVERVIEW.md             live state snapshot
├── web/                    Vite + React 19 + TS + Tailwind v4 + R3F + drei + postprocessing
├── edge/                   Vercel edge functions — Hono, R2, Anthropic, Modal relay
├── pipeline/               Modal app — @app.asgi_app FastAPI + all 4 vision models
└── physics/                standalone Newtonian fit module (pure Python)
```

`physics/` is a pure-Python library the pipeline imports for the fit stage. `web/`, `edge/`, and `pipeline/` deploy independently. Local dev needs `modal setup` + `vercel dev` + `npm run dev`.

---

## Model choices (per stack.md)

- **Segmentation** — SAM 2.1 (Meta, Apache 2.0). Auto-download weights on first run; cached in Modal Volume.
- **Tracking** — CoTracker3 offline (Meta), grid-seeded, ~64 points. Loads via `torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")`.
- **Depth** — Depth Anything V2 Small via HuggingFace `depth-anything/Depth-Anything-V2-Small-hf` with a temporal-smoothing pass. Upgrade path to Video Depth Anything documented in `stages/depth.py`.
- **Scene 4D** — VGGT (`facebook/VGGT-1B`). Camera pose + dynamic scene point cloud from raw video.
- **Physics fit** — `kepler_physics.fit` (already implemented). scipy LM with drag-free closed-form seed + soft simplicity prior.
- **Verdict LLM** — Claude Sonnet 4.5 via Anthropic SDK, streamed through Vercel edge as SSE.

---

## Ports (local dev)

- Web dev server: `http://localhost:5174/`
- Edge dev server (`vercel dev`): `http://localhost:3000/`
- Modal serve (`modal serve pipeline/modal_app.py`): assigned URL

---

## What each dev tool needs

| Tool | Purpose | First-time cost |
|---|---|---|
| **Modal** | GPU pipeline host | `pip install modal && modal setup` (OAuth). Free $30 credit. |
| **Vercel** | Edge functions + web hosting | `npm i -g vercel && vercel login`. Free tier. |
| **Cloudflare R2** | Object storage | Sign up + create bucket + generate S3-compatible access keys. Free tier. |
| **Anthropic Console** | Claude Sonnet 4.5 | Sign up + generate API key. Pay-per-token, ~$0.006 per verdict. |

Total out-of-pocket for demo scale: **~$0** (fits within Modal credit).

---

## Cost per analysis (deployed)

- Modal L4 GPU × ~30 s per analysis @ $0.80/hr = **~$0.007**
- Anthropic Sonnet 4.5 verdict (~500 in, ~300 out tokens) = **~$0.006**
- R2 storage + egress = **~$0**
- Total marginal cost: **~$0.013 per analysis**

Sellable at $1–5 per analysis to newsrooms / legal → 75–380× margin.

---

## Status

Day 1 scaffold shipped. Day 2 = the ambitious version (this document): Modal + Vercel edge + real vision models + cinematic viewer. Three parallel workstreams underway.
