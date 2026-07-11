# YASBM

**Every generated video breaks a law of physics. YASBM proves which one, and where.**

A browser-based forensic tool that takes a short video clip, reconstructs the scene in 3D, tracks rigid objects, fits a Newtonian trajectory to each, and measures where observed motion diverges from what physics would produce. Real footage passes cleanly. AI-generated footage — Sora 2, Veo 3.1, Kling 3.0 — leaves a physics residual you can visualize, quantify, and point at.

The output is a rotating 3D reconstruction with two trajectories overlaid: what the object *did* (red) and what the object *should have done* under Newton's laws given the same release conditions (green). Where they diverge is where the model lied.

---

## Why physics instead of pixel-space detection

Every pixel-artifact deepfake detector dies the moment a new generator ships. Physics doesn't drift. Sora 2 launched with "improved physics simulation" as a headline feature *because it's still what generative video gets most obviously wrong* — bouncing balls that don't conserve momentum, phones that curve mid-air, water that pre-splashes. YASBM sidesteps the arms race by asking a question no future generator can retrain past: **could this have happened in a Newtonian universe?**

---

## Features

- **3D scene reconstruction** — CoTracker3 dense point tracking + Depth Anything V2 monocular depth on an L4 GPU. First frame's RGB is back-projected through its depth into a colored point cloud.
- **Physics fit** — `scipy.optimize.least_squares` LM refinement seeded by a drag-free closed-form fit. Recovers initial velocity, gravity vector, and linear drag coefficient. Reports per-frame residual magnitude in σ units above the empirical noise floor.
- **Cinematic viewer** — React Three Fiber with EffectComposer (Bloom + ACES tone mapping + Vignette). Colored point cloud + emissive trajectory lines + auto-framed cinematic camera on the trajectory bounding box.
- **LLM verdict card** — Claude Sonnet 4.5 streams a plain-English verdict via SSE. Physics-only language, never claims "AI-generated," enumerates 2–3 alternative explanations (off-camera contact force, hidden support, depth artefact, etc.) with probability language.
- **Residual timeline** — SVG line chart of σ over time with a 3σ threshold marker. Peak σ pill in the corner reads *clean* / *borderline* / *flagged*.
- **Verdict**: `< 3σ` = "Physically consistent" / `3–10σ` = "Borderline" / `> 10σ` = "Physically implausible".

---

## Architecture

```
┌─── BROWSER ────────────────────────────────────────────┐
│  Vite + React 19 + TS + Tailwind + R3F + drei          │
│  ├─ Upload / video scrubber                             │
│  ├─ R3F viewer  (point cloud + trajectories)            │
│  │    with @react-three/postprocessing (Bloom + ACES)   │
│  ├─ Residual timeline chart                             │
│  └─ Verdict card (streamed from Claude Sonnet 4.5)      │
└──────────────┬──────────────────────────────────────────┘
               │ HTTPS + SSE for progress
┌─── GPU (Modal, L4 24GB, @app.asgi_app FastAPI) ───────┐
│  Stage 1  SAM 2.1              masks per object        │
│  Stage 2  CoTracker3           dense 2D point tracks   │
│  Stage 3  Depth Anything V2    temporally-stable depth │
│  Stage 4  scene                camera pose + cloud     │
│  Stage 5  Lift 2D→3D           world-frame trajectories│
│  Stage 6  Newton fit           kepler_physics.fit      │
│  Stage 7  Package              PLY + tracks + residuals│
│  Stage 8  Verdict              Claude Sonnet 4.5 SSE   │
└────────────────────────────────────────────────────────┘
```

Everything else is optional:
- **Vercel + Hono edge** — R2 uploads, Modal relay, Anthropic proxy (in `edge/` — deploy if you want per-user rate limits / auth / R2 archival).
- **Cloudflare R2** — video + PLY object storage (skip for the browser-direct-to-Modal path).

---

## Repo layout

```
YASBM/
├── README.md, PROJECT.md, DEPLOY.md, OVERVIEW.md    docs
├── web/           Vite + React 19 + R3F + drei + postprocessing
├── edge/          Hono on Vercel — R2 uploads, Modal relay, Claude verdict
├── pipeline/      Python 3.12 — FastAPI locally, Modal for GPU deploys
└── physics/       Standalone Newtonian fit module (pytest-covered)
```

**Internal module names still use `kepler_pipeline` / `kepler_physics`** — the code hasn't been renamed yet. See the TODO section below.

---

## Quick start (local, browser → Modal)

Two accounts + two terminals.

### 1. Deploy the pipeline to Modal (once)

```powershell
# Prereqs: uv installed at %USERPROFILE%\.local\bin
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
cd pipeline
uv sync
uv run modal setup                              # OAuth
uv run modal secret create yasbm-anthropic ANTHROPIC_API_KEY=sk-ant-...
uv run modal deploy modal_app.py
```

Modal prints a URL like `https://<username>--kepler-pipeline-web.modal.run`. Save it.

### 2. Point the web at Modal

Copy `web/.env.local.example` → `web/.env.local` and set:

```
VITE_PIPELINE_URL=https://<username>--kepler-pipeline-web.modal.run
```

### 3. Run the web dev server

```powershell
cd web
npm install
npm run dev
```

Open <http://localhost:5174/>. Drop a video. Expect ~10–20 s for a warm request (~60–90 s on the very first request while CoTracker3 + Depth Anything V2 weights download into Modal Volumes).

### Fully local (no Modal, stub models)

If you want to iterate on the frontend without hitting the GPU:

```powershell
cd pipeline
uv run uvicorn kepler_pipeline.app:app --host 127.0.0.1 --port 8001 --reload
```

Then set `VITE_PIPELINE_URL=http://127.0.0.1:8001`. Stubs return synthetic straight-line tracks so the whole viewer + residual chart + verdict card lights up in ~2 seconds.

### Physics module tests

```powershell
cd physics
uv run pytest -v
# 4 tests: clean projectile ~0σ, projectile+noise <5σ, teleport >5σ, static <3σ.
```

See [DEPLOY.md](DEPLOY.md) for the full production deploy (Modal + Vercel edge + R2 + Anthropic).

---

## Tech stack

**Browser** — Vite 6, React 19, TypeScript, Tailwind v4 (`@tailwindcss/vite`), Zustand, `@react-three/fiber` v9 + `@react-three/drei` v10 + `@react-three/postprocessing`, custom point cloud shader.

**GPU pipeline** — Python 3.12 via uv, FastAPI + uvicorn locally, `@modal.asgi_app` in production, torch 2.5 + torchvision + opencv-python-headless, transformers, Ultralytics YOLO-World, CoTracker3 (`torch.hub`), Depth Anything V2 Small (Hugging Face), SAM 2.1 (`facebookresearch/sam2`), numpy, scipy.

**Physics** — pure numpy + scipy. `scipy.integrate.solve_ivp` (RK45) or closed-form drag-free integration; `scipy.optimize.least_squares` Levenberg–Marquardt.

**Verdict LLM** — Claude Sonnet 4.5 via the Anthropic Python SDK, streamed as Server-Sent Events.

**Edge (optional)** — Hono on Vercel Node runtime, `@aws-sdk/client-s3` for R2 (S3-compatible), Zod for env validation.

---

## Physics fit — the actual math

For each rigid-object trajectory `p(t_i)` in world coordinates:

- Free parameters: `v0` (3), `g` (3), linear drag `k` (1) → 7 total.
- Model: `x(t) = x0 + ∫ v(t) dt` where `a(t) = g − k · v(t)`.
- Objective: `Σ ‖p(t_i) − sim(v0, g, k, t_i)‖²`.
- Seed: closed-form drag-free least squares on `x(t) = x0 + v0·t + ½·g·t²`. Only accept the nonlinear refit if it beats the linear RMSE by ≥ 30% (soft simplicity prior — prevents the LM solver from wandering into a degenerate high-drag / high-v0 valley on noisy static data).
- Residual per frame: `‖observed − predicted‖`.
- Noise floor: median magnitude of the second-difference of positions across interior samples (robust proxy for measurement noise).
- **Reported σ** = `max(residual) / max(noise_floor, ε)`.

Peak σ becomes `verdict_score`; the LLM turns it into prose.

---

## Verdict-card guardrails

The Anthropic system prompt hard-codes these rules:

1. **Never** claim the video "is AI-generated" or name a specific generator. There is no provenance evidence — only physics.
2. Use exactly one of three verdicts, keyed on σ: *Physically consistent* / *Borderline* / *Physically implausible*.
3. Always cite the specific peak σ, the frame time it occurred at, and the max delta in metres.
4. Enumerate 2–3 alternative explanations with probability language ("more likely / possible / unlikely"): off-camera contact force, hidden support (wire, magnet, rig), occluded interaction, rolling-shutter or motion-blur artefact, unmodelled aerodynamic effect, depth-estimation error.
5. Style: neutral forensic. 60–120 words. No markdown, no bullet points.

---

## Roadmap

**Shipped**
- [x] End-to-end analyze on Modal L4 with CoTracker3 + Depth Anything V2 + physics fit
- [x] Colored point cloud (PLY with per-vertex RGB) from RGB + depth
- [x] Cinematic R3F viewer with EffectComposer + auto-framed camera
- [x] SVG residual timeline with 3σ threshold
- [x] Streamed Claude Sonnet 4.5 verdict card
- [x] Physics module with pytest-covered adversarial cases

**Next**
- [ ] **Rename `kepler_*` modules to `yasbm_*`** — code still uses the old project name internally.
- [ ] **VGGT for real camera pose recovery** — current `scene.py` assumes an identity pose per frame (works for static shots; wrong for anything else).
- [ ] **Audio-visual physics sync** — extend the physics thesis to audio. Cross-correlate predicted contact events (residual/velocity minima) with audio peaks in the accompanying track. Feet-arrive-before-step, ball-bounces-before-contact, splash-pre-empts-water etc. are the same physics violation on a different sense. This absorbs the audio separation + sound-class detection layer from HEED (`../cuhacks`).
- [ ] **Real/fake benchmark set** — curated pairs of real phone footage + Sora / Veo / Kling generations of the same prompt, with expected σ ranges. Currently we test on ad-hoc synthetic clips.
- [ ] **Modal `min_containers=1` on the pipeline class during demo windows** — trades ~$0.80/hr for zero cold-start latency.
- [ ] **N8AO ambient occlusion** — currently disabled because it fights with Bloom's depth-stencil handling on Chrome's WebGL2 driver. Restore once postprocessing decouples the depth attachment.
- [ ] **Video-frame ghost backdrop** — sample the current playhead frame to a canvas texture rendered as a large plane behind the point cloud (opacity by distance).
- [ ] **Provenance metadata check** — read C2PA manifests if present; note the presence/absence in the verdict card without treating either as dispositive.

---

## Cost per analysis (deployed)

| Component | Per-analysis cost |
|---|---|
| Modal L4 GPU (~10 s warm) | ~$0.002 |
| Anthropic Sonnet 4.5 verdict (~500 in, ~300 out tokens) | ~$0.006 |
| R2 storage + egress (if using the edge layer) | ~$0 |
| Vercel edge functions (if using) | $0 (free tier) |
| **Total marginal cost** | **~$0.008** |

Modal's $30 free credit covers ~3,750 analyses.

---

## Security notes

- `.env`, `.env.local`, `.env.*.local` and `.modal.toml` are gitignored. Never commit an Anthropic key, R2 secret, or Modal token.
- The Anthropic key lives on Modal as a named `Secret` (`yasbm-anthropic` / previously `kepler-anthropic`) — the browser never sees it.
- Model weights (`*.pt`, `*.pth`, `*.onnx`, `*.safetensors`) are gitignored and downloaded at runtime into persistent Modal Volumes.

---

## License

TBD. Public preview.
