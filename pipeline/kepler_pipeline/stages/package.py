"""Serialise pipeline outputs into browser-consumable artifacts.

Writes:
    - ``point_cloud.ply`` — ASCII PLY (with per-vertex colour when supplied).
    - ``tracks.json`` — list of ``Track`` dicts (schema.Track).
    - ``residuals.json`` — list of per-frame ``Residual`` dicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_ascii_ply(
    points: np.ndarray,
    colors: np.ndarray | None,
    faces: np.ndarray | None,
    path: Path,
) -> None:
    """Write an ASCII PLY with vertex positions, optional RGB, optional faces."""

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    has_color = colors is not None and colors.size > 0

    if has_color:
        rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if rgb.shape[0] != pts.shape[0]:
            has_color = False

    has_faces = faces is not None and faces.size > 0
    if has_faces:
        fc = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    else:
        fc = np.zeros((0, 3), dtype=np.int32)

    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {pts.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header_lines.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    if has_faces:
        header_lines.append(f"element face {fc.shape[0]}")
        header_lines.append("property list uchar int vertex_indices")
    header_lines.append("end_header\n")
    header = "\n".join(header_lines)

    with path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        if has_color:
            for (x, y, z), (r, g, b) in zip(pts, rgb):
                fh.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for x, y, z in pts:
                fh.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        if has_faces:
            for a, b, c in fc:
                fh.write(f"3 {int(a)} {int(b)} {int(c)}\n")


def package(
    point_cloud_xyz: np.ndarray,
    tracks_3d: np.ndarray,
    timestamps: np.ndarray,
    residuals: dict,
    out_dir: Path,
    point_cloud_rgb: np.ndarray | None = None,
    point_cloud_faces: np.ndarray | None = None,
) -> dict[str, str]:
    """Serialise pipeline outputs into ``out_dir``.

    Parameters
    ----------
    point_cloud_xyz:
        ``(N, 3)`` scene point cloud in metres.
    tracks_3d:
        ``(T, N_tracks, 3)`` lifted world-frame trajectories.
    timestamps:
        ``(T,)`` sample times in seconds.
    residuals:
        Output of the ``physics`` stage — must contain ``residuals``,
        ``per_frame_max`` and ``per_frame_sigma`` (see stages.physics.physics).
    out_dir:
        Directory to write artifacts into. Created if missing.
    point_cloud_rgb:
        Optional ``(N, 3)`` uint8 per-vertex colour. Written into the PLY
        as ``property uchar red/green/blue`` when provided.

    Returns
    -------
    dict[str, str]
        Absolute POSIX paths for ``point_cloud``, ``tracks``, ``residuals``.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ply_path = out_dir / "point_cloud.ply"
    tracks_path = out_dir / "tracks.json"
    residuals_path = out_dir / "residuals.json"

    _write_ascii_ply(
        point_cloud_xyz, point_cloud_rgb, point_cloud_faces, ply_path
    )

    # Tracks JSON: list of Track dicts.
    tracks_payload: list[dict] = []
    if tracks_3d.ndim == 3:
        t, n, _ = tracks_3d.shape
        for track_id in range(n):
            points = [
                {
                    "t_s": float(timestamps[frame_idx]),
                    "position": [
                        float(tracks_3d[frame_idx, track_id, 0]),
                        float(tracks_3d[frame_idx, track_id, 1]),
                        float(tracks_3d[frame_idx, track_id, 2]),
                    ],
                }
                for frame_idx in range(t)
            ]
            tracks_payload.append(
                {
                    "track_id": track_id,
                    "label": f"point_{track_id}",
                    "points": points,
                }
            )

    with tracks_path.open("w", encoding="utf-8") as fh:
        json.dump(tracks_payload, fh, indent=2)

    # Residuals JSON: prefer per-frame arrays produced by the real physics
    # stage; fall back to averaging across tracks for older stub output.
    per_frame_max = residuals.get("per_frame_max") if isinstance(residuals, dict) else None
    per_frame_sigma = (
        residuals.get("per_frame_sigma") if isinstance(residuals, dict) else None
    )
    residuals_payload: list[dict] = []
    if per_frame_max is not None and per_frame_sigma is not None:
        for i in range(int(per_frame_max.shape[0])):
            residuals_payload.append(
                {
                    "t_s": float(timestamps[i]),
                    "delta_m": float(per_frame_max[i]),
                    "sigma": float(per_frame_sigma[i]),
                }
            )
    else:
        residual_arrays = (
            residuals.get("residuals", []) if isinstance(residuals, dict) else []
        )
        if residual_arrays:
            stacked = np.stack(
                [np.asarray(r, dtype=np.float32) for r in residual_arrays], axis=0
            )
            mean_per_frame = stacked.mean(axis=0)
            peak_sigma = float(residuals.get("peak_sigma", 0.0))
            for i in range(mean_per_frame.shape[0]):
                residuals_payload.append(
                    {
                        "t_s": float(timestamps[i]),
                        "delta_m": float(mean_per_frame[i]),
                        "sigma": peak_sigma,
                    }
                )

    with residuals_path.open("w", encoding="utf-8") as fh:
        json.dump(residuals_payload, fh, indent=2)

    return {
        "point_cloud": ply_path.as_posix(),
        "tracks": tracks_path.as_posix(),
        "residuals": residuals_path.as_posix(),
    }
