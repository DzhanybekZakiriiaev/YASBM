"""Camera pose + colored scene point cloud.

Real integration target: **VGGT** (Meta, 2025) or MonST3R for full 4D
reconstruction with recovered camera pose. Until that ships, we use a
practical shortcut:

- Assume a static-ish camera (identity pose per frame).
- Build a coloured point cloud by back-projecting the first frame's RGB
  through its depth map via a pinhole model.

This is not a full 4D reconstruction but it *is* geometrically consistent
with the depth map + trajectories — the point cloud aligns with what
happens in the video, which is all we need for the cinematic viewer.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class SceneOutput(TypedDict):
    camera_poses: np.ndarray  # (T, 4, 4) float32
    xyz: np.ndarray  # (N, 3) float32 world-frame points, metres
    rgb: np.ndarray  # (N, 3) uint8 per-vertex color


def _default_intrinsics(width: int, height: int) -> np.ndarray:
    """Pinhole approximation: focal length ≈ frame width."""

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


def scene(
    frames: list[np.ndarray],
    depth_maps: np.ndarray | None = None,
    max_points: int = 12_000,
) -> SceneOutput:
    """Return camera poses + a colored point cloud.

    Parameters
    ----------
    frames:
        List of RGB uint8 frames ``(H, W, 3)``.
    depth_maps:
        ``(T, H, W)`` per-frame depth in metres. If ``None`` the cloud
        collapses to a small synthetic set (stub compatibility).
    max_points:
        Target upper bound on point count. The frame is subsampled by
        stride to stay under this budget.
    """

    t = len(frames)
    if t == 0:
        return SceneOutput(
            camera_poses=np.zeros((0, 4, 4), dtype=np.float32),
            xyz=np.zeros((0, 3), dtype=np.float32),
            rgb=np.zeros((0, 3), dtype=np.uint8),
        )

    identity = np.broadcast_to(np.eye(4, dtype=np.float32), (t, 4, 4)).copy()

    if depth_maps is None or depth_maps.size == 0:
        # No depth → return a small synthetic cloud so the viewer isn't empty.
        rng = np.random.default_rng(0)
        xyz = rng.random((512, 3), dtype=np.float32) * np.array(
            [2.0, 2.0, 4.0], dtype=np.float32
        ) - np.array([1.0, 1.0, 2.0], dtype=np.float32)
        rgb = np.full((512, 3), 128, dtype=np.uint8)
        return SceneOutput(camera_poses=identity, xyz=xyz, rgb=rgb)

    frame = frames[0]  # (H, W, 3)
    depth = depth_maps[0]  # (H, W)
    height, width = depth.shape

    # Choose a stride so we end up with <= max_points.
    target = max(int(np.ceil(np.sqrt(max_points))), 32)
    stride = max(1, min(height, width) // target)

    intrinsics = _default_intrinsics(width, height)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    grid_v, grid_u = np.meshgrid(ys, xs, indexing="ij")
    grid_v = grid_v.reshape(-1)
    grid_u = grid_u.reshape(-1)

    z = depth[grid_v, grid_u].astype(np.float32)
    x = (grid_u - cx) * z / fx
    y = -(grid_v - cy) * z / fy  # flip Y so up is +Y in world frame

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    rgb = frame[grid_v, grid_u].astype(np.uint8)

    return SceneOutput(camera_poses=identity, xyz=xyz, rgb=rgb)
