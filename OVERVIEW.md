# YASBM — State Snapshot

*Last updated 2026-07-14. One-liner: physics-based plausibility auditing for short videos — real footage obeys Newton, generated footage doesn't.*

Docs map: [README.md](README.md) quick start · [PROJECT.md](PROJECT.md) original pitch · [RESEARCH-V2.md](RESEARCH-V2.md) **v2 architecture decision (read this first)** · [DEPLOY.md](DEPLOY.md) production setup.

---

## ⚠ Operational state: API keys revoked (2026-07-14)

All external API keys were intentionally revoked. Consequences, verified in code:

| Capability | Status without keys |
|---|---|
| `POST /analyze` — full vision pipeline (CoTracker3, Depth Anything V2, YOLOv8, physics, mesh, props placement) | **Fully live** — model weights are cached in Modal Volumes; no external APIs touched |
| 3D viewer, timeline, object chips, hologram props | **Fully live** — static frontend |
| Claude verdict prose (`/verdict`) | Degrades gracefully → deterministic σ-based copy renders instead |
| Poly Pizza GLB retrieval | Returns `glb_url: null` → hologram annotations render (by design) |
| Tripo "3d scan" (`/hero`) | Returns 503 → chip shows "unavailable" (by design) |
| New Modal deploys | Need the Modal token (`modal setup`) restored |

Restore everything with one command once new keys exist:
`uv run modal secret create kepler-anthropic ANTHROPIC_API_KEY=<new> POLYPIZZA_API_KEY=<new> TRIPO_API_KEY=<new>`

---

## What is built and deployed (v1, commit `622b209` + research `bff5958`)

**Pipeline (Modal L4, `pipeline/modal_app.py`)** — stage order: track → depth → lift → physics → objects → exclude-masks → scene → props → package:

- **CoTracker3** (torch.hub) — 64 dense image-plane tracks per clip
- **Depth Anything V2 Small** (HF) — per-frame depth, median-fused
- **YOLOv8n object audit** (`stages/objects.py`) — per-frame detection, greedy-IoU cross-frame identity, track-to-object membership; per-object ballistic fit on centroid; **self-propelled classes (person, car, dog…) exempt** from ballistic verdicts; morph detection = class flicker + 3D rigidity CV + bbox jerk; verdicts: consistent / borderline / implausible / agent / static / morphing
- **Scene reconstruction** (`stages/scene.py`) — median-RGB background plate that ignores object pixels (background fills in where objects moved), YOLO-box exclusion masks keep people out of the static mesh, edge-filtered triangulation, per-frame **dynamic point clouds** of moving-object pixels (the "4D" layer)
- **Props placement** (`stages/props.py`) — per-object 3D position + metric scale from box-center back-projection; object crops saved as artifacts; Poly Pizza GLB resolution (keyed); `/hero` endpoint: crop → Tripo image-to-3D → textured GLB (keyed)
- **Physics core** (`physics/kepler_physics`) — LM ballistic fit with closed-form seed + simplicity prior; 4 pytest cases (clean ≈0σ, teleport ≈22σ)
- **Verdict** (`/verdict`) — Claude Sonnet 4.5 SSE stream over structured evidence, hard-ruled to never claim "AI-generated"

**Frontend (`web/`, Vite + React 19 + R3F)**:

- Video player with per-frame **labeled boxes** color-coded by verdict, following the playhead
- 3D viewer: room mesh + per-frame dynamic points scrubbing with the timeline + verdict-tinted **hologram annotations** at each object's measured position/size (solid models only when a real GLB exists) + "props on/off" toggle
- Camera spawns at the recording camera's exact pose (origin, FOV derived from frame dims) — the scene opens framed like the video
- Scrubbable σ **timeline** (click-to-seek, hover tooltips) synced to video + 3D
- Object chips with per-object σ / morph / verdict + "3d scan" buttons

**Known v1 weaknesses** (what motivated v2): thresholds (3σ/10σ) are uncalibrated guesses; depth-axis noise creates false positives on real clips; whole-trajectory fits ignore that objects are only ballistic in free flight; no real/fake benchmark ever run.

---

## The v2 decision (research complete, build not started)

A two-agent adversarial research round (red-team critic vs systems architect, with a cross-critique round) converged on the **"D3-Anchored Hybrid"** — full reasoning and citations in [RESEARCH-V2.md](RESEARCH-V2.md). Essence:

1. **Physics must be measured on the image plane** — monocular depth jitter produces acceleration noise comparable to gravity (2.8–11 m/s²); CoTracker pixel tracks are ~100× more precise; a ballistic arc is still a parabola in pixels
2. **Primary detector = D3-style second-order image-plane statistics** (training-free, 98.46% mAP on GenVideo, code released) + existing morph signals — fires on every clip
3. **Physics evidence = per-segment image-plane parabola fits** with parameter-consistency tests (gravity direction agreement across segments/objects), not goodness-of-fit
4. **3D scene + ghost trajectory = explainability only**, ghost rendered from the fitted curve, labeled "illustrative"
5. **Calibration on real datasets** (Physics-IQ reals vs VideoPhy-2 fakes, identically re-encoded, bootstrap CIs, ~$10) replaces invented thresholds
6. LLM stays as evidence-grounded explainer (VLMs score ~50% on physical reasoning benchmarks — never the judge)

**Build estimate ≈ 2.5–3.5 days: D3 integration (1d) → image-plane physics refactor (1d) → presentation rework (0.5d) → calibration run (0.5d).** D3 + physics + calibration require no external API keys — only Modal deploy access.

---

## Abandoned / parked (with reasons)

- **ViewCrafter** novel-view flythroughs — 5 failed CUDA image builds, wrong tool for interactive 4D; file parked at `pipeline/viewcrafter_app.py`, frontend behind unset env flag
- **Solid procedural prop models** — clipped through the real mesh, unknowable yaw; replaced by holograms (recoverable at commit `38370db`)
- **Vercel edge layer** (`edge/`) — superseded by direct browser→Modal; kept as reference for auth/rate-limiting later
- **Metric-3D dynamics, TSDF-for-physics, forward-integrated ghosts, VLM-as-judge** — killed by research, reasons in RESEARCH-V2.md §3
- **HEED** (sibling project in `../cuhacks`) — deprioritized entirely; its audio tech is a possible future YASBM feature (audio-visual physics sync)

## Cost & infra notes

- Modal spend limit was hit once (failed GPU image builds + an always-warm container). Both `min_containers=1` were removed; GPU scales to zero after 5 min idle. For demo day: flip `min_containers=1` on 30 min before, off after.
- Marginal cost ~$0.13/analysis with LLM verdict, ~$0.03 without.
