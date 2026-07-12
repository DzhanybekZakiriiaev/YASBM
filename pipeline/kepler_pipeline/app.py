"""FastAPI entrypoint for the KEPLER inference pipeline (local dev).

For production the pipeline runs on Modal via ``modal_app.py``. This
file gives a lightweight local server that exercises the same stage
graph — with stub or CPU-tiny models — for iteration without paying
for a GPU.

Serves ``/health`` and ``/analyze``. Artifacts written by ``package``
are exposed via the ``/artifacts`` static mount so the browser can
fetch PLY / JSON directly.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schema import (
    AnalyzeResponse,
    AnalyzeStatus,
    Residual,
    Track,
    TrajectoryPoint,
)
from .stages.depth import depth as depth_stage
from .stages.lift import lift as lift_stage
from .stages.objects import objects as objects_stage
from .stages.package import package as package_stage
from .stages.physics import physics as physics_stage
from .stages.scene import scene as scene_stage
from .stages.segment import segment as segment_stage
from .stages.track import track as track_stage

MAX_FRAMES = 30

# Root directory that /artifacts is served from. Created at import time so
# the StaticFiles mount always has an existing directory to bind against.
_ARTIFACT_ROOT = Path(tempfile.gettempdir()) / "kepler-pipeline-artifacts"
_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="KEPLER pipeline (local)", version="0.0.1")

# Dev-only permissive CORS so the Vite dev server on :5174 can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/artifacts",
    StaticFiles(directory=str(_ARTIFACT_ROOT)),
    name="artifacts",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "kepler-pipeline", "mode": "local"}


def _read_frames(path: Path, max_frames: int = MAX_FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    finally:
        cap.release()
    return frames


def _default_intrinsics(width: int, height: int) -> np.ndarray:
    fx = fy = float(width)
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, video: UploadFile = File(...)) -> AnalyzeResponse:
    if video.content_type and not video.content_type.startswith("video/"):
        if video.content_type not in ("application/octet-stream", "application/x-mpeg"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content type: {video.content_type}",
            )

    request_id = uuid.uuid4().hex
    request_artifacts = _ARTIFACT_ROOT / request_id
    request_artifacts.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="kepler-upload-"))
    upload_path = tmp_dir / (video.filename or "upload.mp4")
    try:
        with upload_path.open("wb") as fh:
            shutil.copyfileobj(video.file, fh)

        try:
            frames = _read_frames(upload_path)
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=400, detail=f"Failed to read video: {exc}")

        if not frames:
            raise HTTPException(status_code=400, detail="No frames decoded from upload.")

        height, width = frames[0].shape[:2]
        intrinsics = _default_intrinsics(width, height)

        _ = segment_stage(frames)
        tracks_2d = track_stage(frames, grid_size=8)
        depth_maps = depth_stage(frames)
        scene_out = scene_stage(frames, depth_maps=depth_maps)
        camera_poses = scene_out["camera_poses"]
        point_cloud_xyz = scene_out["xyz"]
        point_cloud_rgb = scene_out["rgb"]

        tracks_3d = lift_stage(
            tracks_2d=tracks_2d,
            depth_maps=depth_maps,
            camera_poses=camera_poses,
            intrinsics=intrinsics,
            frame_size=(width, height),
        )

        cap = cv2.VideoCapture(str(upload_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        timestamps = np.arange(len(frames), dtype=np.float32) / float(fps)

        physics_out = physics_stage(tracks_3d, timestamps)

        object_reports = objects_stage(
            frames=frames,
            tracks_2d=tracks_2d,
            tracks_3d=tracks_3d,
            timestamps=timestamps,
        )

        artifact_paths = package_stage(
            point_cloud_xyz=point_cloud_xyz,
            tracks_3d=tracks_3d,
            timestamps=timestamps,
            residuals=physics_out,
            out_dir=request_artifacts,
            point_cloud_rgb=point_cloud_rgb,
            point_cloud_faces=scene_out.get("faces"),
            dynamic_points=scene_out.get("dynamic"),
        )

        # Normalized 2D pixel positions (u, v) in [0, 1] per track per frame.
        tracks_2d_norm = tracks_2d.copy().astype(np.float32)
        tracks_2d_norm[..., 0] /= float(width)
        tracks_2d_norm[..., 1] /= float(height)
        tracks_2d_norm = np.clip(tracks_2d_norm, 0.0, 1.0)
        per_track_sigma = physics_out.get("per_track_sigma")

        response_tracks: list[Track] = []
        t, n, _ = tracks_3d.shape
        for track_id in range(n):
            sigmas = (
                [float(per_track_sigma[track_id, i]) for i in range(t)]
                if per_track_sigma is not None
                else None
            )
            response_tracks.append(
                Track(
                    track_id=track_id,
                    label=f"point_{track_id}",
                    points=[
                        TrajectoryPoint(
                            t_s=float(timestamps[i]),
                            position=(
                                float(tracks_3d[i, track_id, 0]),
                                float(tracks_3d[i, track_id, 1]),
                                float(tracks_3d[i, track_id, 2]),
                            ),
                        )
                        for i in range(t)
                    ],
                    points_2d=[
                        (
                            float(tracks_2d_norm[i, track_id, 0]),
                            float(tracks_2d_norm[i, track_id, 1]),
                        )
                        for i in range(t)
                    ],
                    sigma_per_frame=sigmas,
                )
            )

        peak_sigma = float(physics_out.get("peak_sigma", 0.0))
        per_frame_max = physics_out.get("per_frame_max")
        per_frame_sigma = physics_out.get("per_frame_sigma")
        response_residuals: list[Residual] = []
        if per_frame_max is not None and per_frame_sigma is not None:
            for i in range(int(per_frame_max.shape[0])):
                response_residuals.append(
                    Residual(
                        t_s=float(timestamps[i]),
                        delta_m=float(per_frame_max[i]),
                        sigma=float(per_frame_sigma[i]),
                    )
                )

        ply_name = Path(artifact_paths["point_cloud"]).name
        base_url = str(request.base_url).rstrip("/")
        point_cloud_url = f"{base_url}/artifacts/{request_id}/{ply_name}"
        dynamic_points_url = f"{base_url}/artifacts/{request_id}/dynamic.json"
        duration_s = float(timestamps[-1]) if len(timestamps) else 0.0

        eligible_sigmas = [
            o["ballistic"]["sigma"] for o in object_reports if o["ballistic"]["eligible"]
        ]
        if eligible_sigmas:
            verdict_score = float(max(eligible_sigmas))
            verdict_basis = "objects"
        else:
            verdict_score = peak_sigma
            verdict_basis = "grid_fallback"

        return AnalyzeResponse(
            status=AnalyzeStatus.done,
            tracks=response_tracks,
            residuals=response_residuals,
            verdict_score=verdict_score,
            verdict_basis=verdict_basis,
            objects=object_reports,
            max_morph_score=max(
                (o["morph_score"] for o in object_reports), default=0.0
            ),
            point_cloud_url=point_cloud_url,
            dynamic_points_url=dynamic_points_url,
            error=None,
            frame_width=int(width),
            frame_height=int(height),
            fps=float(fps),
            duration_s=duration_s,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
