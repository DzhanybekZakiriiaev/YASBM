# kepler-pipeline

Python inference pipeline for KEPLER. Two entry points:

- **`kepler_pipeline/app.py`** — local FastAPI for iteration (stubs fall through when torch is not installed).
- **`modal_app.py`** — Modal deployment. Ships CoTracker3 + Depth Anything V2 + SAM 2.1 + physics on a warm L4 container.

---

## Local dev (CPU, stubs)

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv sync
uv run uvicorn kepler_pipeline.app:app --host 127.0.0.1 --port 8001 --reload
```

Health check:

```powershell
curl http://127.0.0.1:8001/health
# {"status":"ok","service":"kepler-pipeline","mode":"local"}
```

Endpoints:

- `GET  /health`
- `POST /analyze` (multipart, field `video`) → `AnalyzeResponse`
- `GET  /artifacts/{request_id}/{filename}` → served PLY / JSON artifacts

---

## Production — Modal

See the root [`DEPLOY.md`](../DEPLOY.md#1-pipeline--modal) for the full walkthrough. Short version:

```powershell
uv run modal setup                        # once, OAuth
uv run modal deploy modal_app.py          # deploys pipeline to Modal L4
```

Modal prints a `https://<name>-web.modal.run` URL. Health-check it and you're done.

---

## Model integration status

| Stage | Real integration | Stub fallback |
|---|---|---|
| `segment` | SAM 2.1 (`sam2` package) | Empty per-frame mask lists |
| `track` | CoTracker3 offline (`torch.hub`) | Synthetic left-to-right grid sweep |
| `depth` | Depth Anything V2 Small (`transformers`) | Constant Z = 5 m |
| `scene` | Colored point cloud from RGB + depth (identity pose) | Small synthetic cloud |
| `lift` | Real — pinhole back-projection | n/a |
| `physics` | Real — `kepler_physics.fit` | n/a |
| `package` | Real — colored ASCII PLY + JSON | n/a |

Each real-model stage top-level-imports its ML deps inside a `try/except ImportError`. If those imports fail (typical local venv without torch), the stage returns a synthetic result of the correct shape so `app.py` still runs end-to-end. This is what makes the "local iteration in ~2.5 seconds without a GPU" workflow possible.

VGGT for full camera-pose recovery is a follow-up. Current `scene.py` assumes an identity camera pose per frame and builds the colored point cloud from the first-frame RGB back-projected through the first-frame depth.

---

## Modal weight caching

Three named Modal Volumes persist across deploys so cold-start weight downloads are one-time:

- `kepler-torch-hub` mounted at `/root/.cache/torch` — CoTracker3 checkpoint
- `kepler-hf-cache` mounted at `/root/.cache/huggingface` — Depth Anything V2 weights
- `kepler-artifacts` mounted at `/artifacts` — output PLY / JSON files, shared between the GPU pipeline class and the web ASGI function so the web function can serve files the pipeline wrote.

Artifacts are written under `/artifacts/<request_id>/` with `point_cloud.ply`, `tracks.json`, `residuals.json`.
