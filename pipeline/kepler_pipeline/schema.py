"""Pydantic v2 schemas exchanged between the pipeline and the browser."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

Vec3 = tuple[float, float, float]


class TrajectoryPoint(BaseModel):
    """A single sample on an object's 3D trajectory."""

    t_s: float = Field(..., description="Timestamp in seconds from clip start.")
    position: Vec3 = Field(..., description="World-frame (x, y, z) in metres.")


class Track(BaseModel):
    """A per-object 3D trajectory over time."""

    track_id: int
    label: str
    points: list[TrajectoryPoint]


class Residual(BaseModel):
    """Physics-fit residual at a single frame."""

    t_s: float
    delta_m: float = Field(..., description="Distance (m) between observed and fitted position.")
    sigma: float = Field(..., description="Residual normalised by fit uncertainty.")


class AnalyzeStatus(str, Enum):
    pending = "pending"
    segmenting = "segmenting"
    tracking = "tracking"
    depth = "depth"
    scene = "scene"
    lifting = "lifting"
    fitting = "fitting"
    packaging = "packaging"
    done = "done"
    error = "error"


class AnalyzeResponse(BaseModel):
    """Top-level payload returned by ``POST /analyze``."""

    status: AnalyzeStatus
    tracks: list[Track]
    residuals: list[Residual]
    verdict_score: float = Field(
        ...,
        description="Peak physics-violation sigma. Higher = more likely fake.",
    )
    point_cloud_url: str | None = None
    error: str | None = None
