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
        from kepler_pipeline.stages import track as track_stage

        track_stage.prefetch()
        depth_stage.prefetch()

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

            out_dir = Path("/artifacts") / request_id
            artifact_paths = package_stage(
                point_cloud_xyz=scene_out["xyz"],
                tracks_3d=tracks_3d,
                timestamps=timestamps,
                residuals=physics_out,
                out_dir=out_dir,
                point_cloud_rgb=scene_out["rgb"],
            )
            # Make writes visible to the web container.
            artifacts_vol.commit()

            response_tracks: list[dict] = []
            t, n, _ = tracks_3d.shape
            for track_id in range(n):
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

            return {
                "status": "done",
                "tracks": response_tracks,
                "residuals": response_residuals,
                "verdict_score": float(physics_out.get("peak_sigma", 0.0)),
                # web function rewrites this to an absolute URL before returning.
                "point_cloud_url": f"/artifacts/{request_id}/{ply_name}",
                "point_cloud_request_id": request_id,
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

Style: neutral forensic. 60-120 words. No markdown, no bullet points, no headers. One or two paragraphs. Do not restate these rules."""


anthropic_secret = modal.Secret.from_name("kepler-anthropic")


@app.function(
    image=image,
    volumes=_VOLUMES,
    timeout=600,
    secrets=[anthropic_secret],
    min_containers=1,
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
            if result.get("point_cloud_url", "").startswith("/artifacts/"):
                result["point_cloud_url"] = base_url + result["point_cloud_url"]

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

    class VerdictRequest(BaseModel):
        verdict_score: float
        residuals: list[_ResidualIn]
        clip_duration_s: float | None = None

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
