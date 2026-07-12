"""Modal deployment for the KEPLER inference pipeline.

Deploy with::

    modal setup                       # once, OAuth
    modal deploy pipeline/modal_app.py

Modal prints a public URL like ``https://<name>-web.modal.run``. That is
the URL to give the edge layer (``MODAL_PIPELINE_URL`` in ``edge/.env``)
or to point ``VITE_PIPELINE_URL`` at directly for a browser-only setup.

The pipeline runs on a warm L4 container (24 GB VRAM). Vision models are
preloaded via ``@modal.enter()`` so a first request pays only the
inference cost, not the model-load cost. Weights + HF cache persist
between deploys via named Modal Volumes.

**Do NOT** add ``from __future__ import annotations`` at the top of this
module. FastAPI's Pydantic v2 type-adapter can't resolve string
annotations for imports that live inside the ``def web()`` scope, so
``UploadFile`` degrades to an unresolvable ``ForwardRef`` and every
POST /analyze returns 500 before the handler body runs.
"""

import base64
import io
import shutil
import tempfile
import uuid
from pathlib import Path

import modal
from fastapi import File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------
# We install torch + transformers + CoTracker3 inside a debian_slim image.
# The local pipeline code + the sibling physics/ package are copied into
# the container at build time via ``add_local_python_source`` /
# ``add_local_dir``.

_PIPELINE_ROOT = Path(__file__).parent
_PHYSICS_ROOT = _PIPELINE_ROOT.parent / "physics"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "opencv-python-headless>=4.10",
        "transformers>=4.45",
        "accelerate>=0.34",
        "einops",
        "imageio[ffmpeg]",
        "pillow",
        "numpy<3",
        "scipy",
        "fastapi",
        "python-multipart",
        "pydantic>=2.9",
        "huggingface_hub",
        "anthropic>=0.34",
        # Object detection for the object-centric audit (objects.py stage).
        "ultralytics>=8.3",
    )
    .pip_install(
        "git+https://github.com/facebookresearch/co-tracker.git@main",
    )
    .add_local_python_source("kepler_pipeline")
    .add_local_dir(str(_PHYSICS_ROOT / "kepler_physics"), "/root/kepler_physics")
)


# ---------------------------------------------------------------------------
# Modal app + persistent volumes
# ---------------------------------------------------------------------------
app = modal.App("kepler-pipeline")

# torch.hub caches CoTracker3 weights under ~/.cache/torch/hub.
hub_cache = modal.Volume.from_name("kepler-torch-hub", create_if_missing=True)
# Depth Anything weights land under ~/.cache/huggingface.
hf_cache = modal.Volume.from_name("kepler-hf-cache", create_if_missing=True)
# Shared artifact directory so the web function can serve PLY / JSON
# files that the Pipeline class wrote on a different container.
artifacts_vol = modal.Volume.from_name("kepler-artifacts", create_if_missing=True)

_VOLUMES = {
    "/root/.cache/torch": hub_cache,
    "/root/.cache/huggingface": hf_cache,
    "/artifacts": artifacts_vol,
}


# ---------------------------------------------------------------------------
# Pipeline — warm-loaded model container
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu="L4",
    volumes=_VOLUMES,
    timeout=600,
    scaledown_window=300,
)
class KeplerPipeline:
    @modal.enter()
    def load(self) -> None:
        """Warm-load CoTracker3 + Depth Anything V2 so the first analyze
        request pays only inference cost."""

        from kepler_pipeline.stages import depth as depth_stage
        from kepler_pipeline.stages import objects as objects_stage
        from kepler_pipeline.stages import track as track_stage

        track_stage.prefetch()
        depth_stage.prefetch()
        objects_stage.prefetch()

    @modal.method()
    def analyze_bytes(
        self, video_bytes: bytes, request_id: str
    ) -> dict:
        """Run the full pipeline on ``video_bytes``; write artifacts under
        ``/artifacts/{request_id}/``; return the response payload."""

        import cv2
        import numpy as np

        from kepler_pipeline.stages.depth import depth as depth_stage
        from kepler_pipeline.stages.lift import lift as lift_stage
        from kepler_pipeline.stages.objects import objects as objects_stage
        from kepler_pipeline.stages.package import package as package_stage
        from kepler_pipeline.stages.physics import physics as physics_stage
        from kepler_pipeline.stages.scene import scene as scene_stage
        from kepler_pipeline.stages.track import track as track_stage

        tmp_dir = Path(tempfile.mkdtemp(prefix="kepler-mod-"))
        upload_path = tmp_dir / "upload.mp4"
        try:
            upload_path.write_bytes(video_bytes)

            cap = cv2.VideoCapture(str(upload_path))
            frames: list[np.ndarray] = []
            try:
                while len(frames) < 30:
                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        break
                    frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            finally:
                cap.release()

            if not frames:
                return {
                    "status": "error",
                    "tracks": [],
                    "residuals": [],
                    "verdict_score": 0.0,
                    "point_cloud_url": None,
                    "error": "no frames decoded",
                }
            if not np.isfinite(fps) or fps <= 0:
                fps = 30.0
            timestamps = np.arange(len(frames), dtype=np.float32) / float(fps)

            height, width = frames[0].shape[:2]
            fx = fy = float(width)
            intrinsics = np.array(
                [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

            tracks_2d = track_stage(frames, grid_size=8)
            depth_maps = depth_stage(frames)
            scene_out = scene_stage(frames, depth_maps=depth_maps)
            tracks_3d = lift_stage(
                tracks_2d=tracks_2d,
                depth_maps=depth_maps,
                camera_poses=scene_out["camera_poses"],
                intrinsics=intrinsics,
                frame_size=(width, height),
            )
            physics_out = physics_stage(tracks_3d, timestamps)

            # Object-centric audit: named objects, per-object ballistic
            # verdicts (self-propelled classes exempt), morph scores.
            object_reports = objects_stage(
                frames=frames,
                tracks_2d=tracks_2d,
                tracks_3d=tracks_3d,
                timestamps=timestamps,
            )

            out_dir = Path("/artifacts") / request_id
            artifact_paths = package_stage(
                point_cloud_xyz=scene_out["xyz"],
                tracks_3d=tracks_3d,
                timestamps=timestamps,
                residuals=physics_out,
                out_dir=out_dir,
                point_cloud_rgb=scene_out["rgb"],
                point_cloud_faces=scene_out.get("faces"),
                dynamic_points=scene_out.get("dynamic"),
            )
            # Make writes visible to the web container.
            artifacts_vol.commit()

            # Normalized 2D positions (u, v) in [0, 1] per track per frame.
            # tracks_2d shape (T, N, 2) is (u_px, v_px). Divide by frame size.
            tracks_2d_norm = tracks_2d.copy().astype(np.float32)
            tracks_2d_norm[..., 0] /= float(width)
            tracks_2d_norm[..., 1] /= float(height)
            tracks_2d_norm = np.clip(tracks_2d_norm, 0.0, 1.0)

            per_track_sigma = physics_out.get("per_track_sigma")

            response_tracks: list[dict] = []
            t, n, _ = tracks_3d.shape
            for track_id in range(n):
                sigmas_2d = (
                    [
                        float(per_track_sigma[track_id, i])
                        for i in range(t)
                    ]
                    if per_track_sigma is not None
                    else None
                )
                response_tracks.append(
                    {
                        "track_id": track_id,
                        "label": f"point_{track_id}",
                        "points": [
                            {
                                "t_s": float(timestamps[i]),
                                "position": [
                                    float(tracks_3d[i, track_id, 0]),
                                    float(tracks_3d[i, track_id, 1]),
                                    float(tracks_3d[i, track_id, 2]),
                                ],
                            }
                            for i in range(t)
                        ],
                        "points_2d": [
                            [
                                float(tracks_2d_norm[i, track_id, 0]),
                                float(tracks_2d_norm[i, track_id, 1]),
                            ]
                            for i in range(t)
                        ],
                        "sigma_per_frame": sigmas_2d,
                    }
                )

            per_frame_max = physics_out.get("per_frame_max")
            per_frame_sigma = physics_out.get("per_frame_sigma")
            response_residuals: list[dict] = []
            if per_frame_max is not None and per_frame_sigma is not None:
                for i in range(int(per_frame_max.shape[0])):
                    response_residuals.append(
                        {
                            "t_s": float(timestamps[i]),
                            "delta_m": float(per_frame_max[i]),
                            "sigma": float(per_frame_sigma[i]),
                        }
                    )

            ply_name = Path(artifact_paths["point_cloud"]).name
            duration_s = float(timestamps[-1]) if len(timestamps) else 0.0

            # Verdict policy: when recognizable objects exist, the score is
            # the max σ over *eligible* objects (rigid + moving). Blind grid
            # tracks sitting on walls no longer manufacture false positives.
            # Without any recognizable object we fall back to the grid-level
            # peak sigma but note the low confidence for the LLM verdict.
            eligible_sigmas = [
                o["ballistic"]["sigma"]
                for o in object_reports
                if o["ballistic"]["eligible"]
            ]
            if eligible_sigmas:
                verdict_score = float(max(eligible_sigmas))
                verdict_basis = "objects"
            else:
                verdict_score = float(physics_out.get("peak_sigma", 0.0))
                verdict_basis = "grid_fallback"
            max_morph = max(
                (o["morph_score"] for o in object_reports), default=0.0
            )

            return {
                "status": "done",
                "tracks": response_tracks,
                "residuals": response_residuals,
                "verdict_score": verdict_score,
                "verdict_basis": verdict_basis,
                "objects": object_reports,
                "max_morph_score": float(max_morph),
                # web function rewrites these to absolute URLs before returning.
                "point_cloud_url": f"/artifacts/{request_id}/{ply_name}",
                "dynamic_points_url": f"/artifacts/{request_id}/dynamic.json",
                "point_cloud_request_id": request_id,
                "frame_width": int(width),
                "frame_height": int(height),
                "fps": float(fps),
                "duration_s": duration_s,
                "error": None,
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# ASGI web function — public HTTPS endpoint
# ---------------------------------------------------------------------------


_VERDICT_SYSTEM_PROMPT = """You write the verdict card for KEPLER, a physics-based video plausibility auditor.

You are given a peak-sigma score (verdict_score) plus a per-frame residual timeline for a Newtonian projectile fit against a 3D object trajectory recovered from a short video clip.

Hard rules:
1. NEVER claim the video "is AI-generated" or "was made by Sora / Veo / Kling" or any specific generator. You have no evidence of provenance — only physics.
2. Use one of exactly these three verdicts, chosen by verdict_score (peak sigma):
   - "Physically consistent" if verdict_score < 3
   - "Borderline" if 3 <= verdict_score < 10
   - "Physically implausible" if verdict_score >= 10
3. Cite the specific peak sigma value, the frame time it occurred at, and the max delta in metres.
4. Enumerate 2-3 alternative explanations for any flagged violation with probability language ("more likely / possible / unlikely"). Reasonable candidates: an off-camera contact force, a hidden support (wire, magnet, rig), an occluded interaction, a rolling-shutter or motion-blur artefact, an unmodelled aerodynamic effect, a depth-estimation error.
5. If per-object reports are provided, ground the verdict in them: name the objects by their detected labels. Self-propelled objects (persons, vehicles, animals) are EXEMPT from the ballistic check — never cite their trajectory as a violation. If an object's morph_score exceeds 0.6, report a shape-consistency violation ("the object's detected identity or rigid geometry is unstable over time"), which is a separate, strong anomaly signal independent of trajectory physics.
6. If verdict_basis is "grid_fallback", state that no recognizable free-flight object was found and the confidence of the trajectory audit is reduced.

Style: neutral forensic. 60-120 words. No markdown, no bullet points, no headers. One or two paragraphs. Do not restate these rules."""


anthropic_secret = modal.Secret.from_name("kepler-anthropic")


@app.function(
    image=image,
    volumes=_VOLUMES,
    timeout=600,
    secrets=[anthropic_secret],
    # NOTE: no min_containers here. A permanently-warm container bills 24/7
    # and drained the workspace credit once already. For demo day, flip
    # min_containers=1 on ~30 min before, remove right after.
)
@modal.asgi_app()
def web():
    """Public FastAPI endpoint. Deployed at ``<name>-web.modal.run``."""

    import json as _json

    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="KEPLER pipeline (Modal)", version="0.0.1")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "kepler-pipeline", "mode": "modal"}

    @api.get("/artifacts/{req_id}/{filename}")
    async def artifact(req_id: str, filename: str) -> FileResponse:
        # Reload volume so this container sees writes from the GPU container.
        artifacts_vol.reload()
        path = Path("/artifacts") / req_id / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="artifact not found")
        media = (
            "application/octet-stream" if filename.endswith(".ply") else "application/json"
        )
        return FileResponse(str(path), media_type=media)

    @api.post("/analyze")
    async def analyze(request: Request):
        """POST a video; returns AnalyzeResponse dict.

        We parse the multipart form manually via ``request.form()`` instead
        of taking ``video: UploadFile = File(...)`` as a typed parameter.
        FastAPI's Pydantic v2 type adapter can't resolve ``UploadFile`` when
        the handler is defined inside another function (as it must be for
        ``@modal.asgi_app``), and every call bombs with a ``ForwardRef``
        error before the body runs. Manual parsing sidesteps that entirely.
        """
        try:
            form = await request.form()
            video = form.get("video")
            if video is None or not hasattr(video, "read"):
                return JSONResponse(
                    status_code=400,
                    content={"detail": "missing 'video' field in multipart body"},
                )

            body = await video.read()
            if not body:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "empty upload"},
                )

            request_id = uuid.uuid4().hex
            # `.remote.aio()` awaits the remote call properly on the asyncio
            # loop. Bare `.remote()` blocks the loop and Modal's ASGI kills
            # the request with a plain-text 500 (no CORS).
            result = await KeplerPipeline().analyze_bytes.remote.aio(
                body, request_id
            )

            base_url = str(request.base_url).rstrip("/")
            for key in ("point_cloud_url", "dynamic_points_url"):
                if result.get(key, "").startswith("/artifacts/"):
                    result[key] = base_url + result[key]

            return result
        except Exception as exc:
            import traceback

            tb = traceback.format_exc()
            print(f"[/analyze] error: {exc}\n{tb}")
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"{type(exc).__name__}: {exc}",
                    "traceback": tb.splitlines()[-10:],
                },
            )

    class _ResidualIn(BaseModel):
        t_s: float
        delta_m: float
        sigma: float

    class _ObjectIn(BaseModel):
        label: str
        verdict: str
        sigma: float = 0.0
        morph_score: float = 0.0
        reason: str = ""

    class VerdictRequest(BaseModel):
        verdict_score: float
        residuals: list[_ResidualIn]
        clip_duration_s: float | None = None
        verdict_basis: str | None = None
        objects: list[_ObjectIn] | None = None

    @api.post("/verdict")
    async def verdict(body: VerdictRequest):
        """SSE stream of Claude Sonnet 4.5's verdict card.

        The client posts the numeric residuals; we downsample to keep tokens
        cheap, feed them to Claude with a strict physics-plausibility system
        prompt, and stream the text back as ``data: {"type":"token",...}``
        events terminated by a single ``data: {"type":"done"}``.
        """

        import os

        from anthropic import AsyncAnthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY not configured (create the Modal "
                "secret 'kepler-anthropic' with your Anthropic key).",
            )

        client = AsyncAnthropic(api_key=api_key)

        # Downsample residuals: peak + p95 + ~8 evenly-spaced samples.
        residuals = [r.model_dump() for r in body.residuals]
        peak = max(residuals, key=lambda r: r["sigma"]) if residuals else None
        residuals_by_sigma = sorted(residuals, key=lambda r: -r["sigma"])
        p95_idx = min(
            len(residuals_by_sigma) - 1, max(0, int(len(residuals_by_sigma) * 0.05))
        )
        p95 = residuals_by_sigma[p95_idx] if residuals_by_sigma else None
        n_samples = min(8, len(residuals))
        stride = max(1, len(residuals) // max(n_samples, 1))
        samples = residuals[::stride][:n_samples]

        user_msg_parts: list[str] = [
            f"verdict_score (peak sigma): {body.verdict_score:.3f}",
        ]
        if peak:
            user_msg_parts.append(
                f"peak residual frame: t={peak['t_s']:.3f}s, "
                f"sigma={peak['sigma']:.3f}, delta={peak['delta_m']:.4f} m"
            )
        if p95:
            user_msg_parts.append(
                f"p95 residual: t={p95['t_s']:.3f}s, sigma={p95['sigma']:.3f}"
            )
        if body.clip_duration_s is not None:
            user_msg_parts.append(f"clip duration: {body.clip_duration_s:.2f}s")
        if body.verdict_basis:
            user_msg_parts.append(f"verdict_basis: {body.verdict_basis}")
        if body.objects:
            obj_lines = "; ".join(
                f"{o.label} [verdict={o.verdict}, sigma={o.sigma:.2f}, "
                f"morph={o.morph_score:.2f}, note={o.reason or 'n/a'}]"
                for o in body.objects[:8]
            )
            user_msg_parts.append(f"detected objects: {obj_lines}")
        user_msg_parts.append(
            "residual samples (t_s, delta_m, sigma): "
            + ", ".join(
                f"({s['t_s']:.2f}, {s['delta_m']:.4f}, {s['sigma']:.2f})"
                for s in samples
            )
        )
        user_msg = "\n".join(user_msg_parts)

        async def event_stream():
            try:
                async with client.messages.stream(
                    model="claude-sonnet-4-5",
                    max_tokens=400,
                    system=_VERDICT_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                ) as stream:
                    async for text in stream.text_stream:
                        payload = _json.dumps({"type": "token", "text": text})
                        yield f"data: {payload}\n\n"
                yield "data: " + _json.dumps({"type": "done"}) + "\n\n"
            except Exception as exc:  # pragma: no cover
                err = _json.dumps({"type": "error", "message": str(exc)})
                yield f"data: {err}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return api


# ---------------------------------------------------------------------------
# Local smoke test — run with ``modal run pipeline/modal_app.py``.
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def smoke_test() -> None:
    """Synthesize a 30-frame MP4, POST it through the Pipeline, print
    a short summary. Requires Modal auth (``modal setup``)."""

    import cv2
    import numpy as np

    tmp = Path(tempfile.mkdtemp())
    video_path = tmp / "smoke.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    for i in range(30):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = int(20 + (280 * i / 29))
        y = int(50 + 100 * (1 - (i / 29 - 0.5) ** 2 * 4))
        cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    body = video_path.read_bytes()
    result = KeplerPipeline().analyze_bytes.remote(body, uuid.uuid4().hex)
    print(f"status: {result.get('status')}")
    print(f"tracks: {len(result.get('tracks', []))}")
    print(f"residuals: {len(result.get('residuals', []))}")
    print(f"verdict_score: {result.get('verdict_score')}")
    print(f"point_cloud_url: {result.get('point_cloud_url')}")


# Prevent linters from flagging unused imports (kept for optional inline
# base64 export if callers ever need bytes instead of a URL).
_ = base64, io
