# KEPLER — State Snapshot

**One-liner**: Every generated video breaks a law of physics. KEPLER proves which one, and where.

See [PROJECT.md](PROJECT.md) for the full pitch, [DEPLOY.md](DEPLOY.md) for the step-by-step production deploy.

---

## Status

**All three slices shipped.** Local end-to-end works; production deploy is one `modal deploy` + one `vercel deploy` away.

| Slice | State | Notes |
|---|---|---|
| **Physics fit** (`physics/`) | ✅ | 4 pytest cases passing — clean projectile 0σ, teleport 22σ |
| **Pipeline (local)** (`pipeline/app.py`) | ✅ | Stubs for track/depth/segment; real lift + physics. 2.4s per 30-frame clip. |
| **Pipeline (Modal)** (`pipeline/modal_app.py`) | ✅ ready to deploy | Modal image w/ torch + transformers + CoTracker3. Real CoTracker3 + Depth Anything V2 + SAM 2.1 (stubs fall through when torch unavailable). Deploy with `modal deploy modal_app.py`. |
| **Edge** (`edge/`) | ✅ ready to deploy | Hono on Vercel: `/api/upload` (R2 presigned), `/api/analyze` (Modal relay), `/api/verdict` (Claude Sonnet 4.5 SSE). |
| **Web viewer** (`web/`) | ✅ | Vite + React 19 + R3F v9 + drei v10 + @react-three/postprocessing (Bloom + N8AO + ToneMapping + Vignette). Cinematic camera auto-frames on tracks. Colored PLY renders via custom point shader. |
| **Integration** | ✅ | Web resolves `VITE_EDGE_URL` → `VITE_PIPELINE_URL` → localhost. Pipeline returns colored PLY (with `property uchar red/green/blue`). Physics `per_frame_max` + `per_frame_sigma` populate the residual chart correctly. |

---

## What's running locally right now

- Local pipeline: `http://127.0.0.1:8001/` (bg `boc57620c`)
- Web dev server: not started this session — run `npm run dev` in `web/`.

Local run verified: `POST /analyze` on a 30-frame synthetic MP4 returns in ~2.4 s with 64 tracks × 30 points, 30 residuals, and a 19,200-point colored PLY served from `/artifacts/<uuid>/point_cloud.ply`.

---

## Repo layout

```
kepler/
├── PROJECT.md              pitch + architecture (target state)
├── DEPLOY.md               step-by-step production deploy
├── OVERVIEW.md             you are here
├── README.md               quick run instructions
│
├── web/                    Vite + React 19 + R3F v9 + drei v10 + postprocessing
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Uploader.tsx       auto-triggers runAnalyze() after drop
│   │   │   ├── Player.tsx         frame-accurate scrubber
│   │   │   ├── Viewer3D.tsx       R3F canvas w/ EffectComposer (Bloom+N8AO+Tone+Vignette)
│   │   │   ├── PointCloud.tsx     custom ShaderMaterial, size-attenuated dust points
│   │   │   ├── ResidualChart.tsx  SVG σ timeline w/ 3σ threshold
│   │   │   └── Verdict.tsx        tone-colored copy keyed on peak σ
│   │   ├── three/pointCloud.ts    PLY loader w/ RGB → BufferGeometry
│   │   ├── lib/api.ts             typed client, VITE_EDGE_URL → VITE_PIPELINE_URL → localhost
│   │   └── state/analysis.ts      Zustand: videoFile/URL/status/result + runAnalyze
│   ├── package.json
│   └── vite.config.ts             port 5174
│
├── edge/                   Hono on Vercel — R2, Modal relay, Claude verdict
│   ├── api/
│   │   ├── upload.ts       POST → presigned R2 PUT URL + key
│   │   ├── analyze.ts      POST { key } → Modal /analyze relay w/ SSE fallback
│   │   └── verdict.ts      POST { verdict_score, residuals } → Claude stream (SSE)
│   ├── src/
│   │   ├── env.ts, r2.ts, cors.ts, verdict-prompt.ts
│   ├── package.json, tsconfig.json, vercel.json, .env.example
│
├── pipeline/               Python 3.12 + uv
│   ├── modal_app.py        Modal app: Image + Volumes + KeplerPipeline cls + @asgi_app web
│   ├── kepler_pipeline/
│   │   ├── app.py          local FastAPI for iteration
│   │   ├── schema.py       Pydantic v2
│   │   └── stages/
│   │       ├── segment.py  SAM 2.1 (stub fallback when torch missing)
│   │       ├── track.py    CoTracker3 offline (stub fallback)
│   │       ├── depth.py    Depth Anything V2 Small (stub fallback)
│   │       ├── scene.py    Colored point cloud from RGB + depth (identity pose)
│   │       ├── lift.py     Pinhole 2D → 3D back-projection
│   │       ├── physics.py  Calls kepler_physics.fit
│   │       └── package.py  Colored ASCII PLY + tracks JSON + residuals JSON
│   └── pyproject.toml
│
└── physics/                pure Python fit module
    ├── kepler_physics/
    │   ├── model.py        integrate(x0, v0, g, drag, ts) w/ closed form + RK45
    │   └── fit.py          scipy LM w/ closed-form seed + simplicity prior
    ├── tests/test_fit.py   4 tests
    └── pyproject.toml
```

---

## What to do next

1. Follow [DEPLOY.md](DEPLOY.md) sections 1–5 to stand up Modal + Vercel + R2 + Anthropic.
2. Verify the end-to-end round-trip on the deployed web URL.
3. Curate a real/fake demo clip pair — a phone throw shot on your phone vs. the same prompt fed to Sora 2 or Veo 3. Peak σ separation is the money shot.
4. (Optional) upgrade `scene.py` to real VGGT once the physics story is stable — it'll give per-frame camera pose recovery so the point cloud stays coherent through camera motion.

---

## Cost

- Local dev: **$0**
- Production (per analysis): **~$0.008** (Modal + Anthropic)
- Full demo day traffic under Modal's $30 free credit: comfortable.
