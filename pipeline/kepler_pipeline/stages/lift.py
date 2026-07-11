"""Lift 2D pixel tracks + depth + camera pose into 3D world coordinates.

This stage is a real implementation, not a stub. Given a 2D track sample
``(u, v)`` at frame ``t`` with metric depth ``Z``, we invert the pinhole
projection:

.. code::

    X_c = (u - c_x) * Z / f_x
    Y_c = (v - c_y) * Z / f_y
    Z_c = Z

producing a camera-frame point ``P_c = (X_c, Y_c, Z_c)``. We then apply the
camera-to-world transform ``T_wc`` (the pose matrix provided by the scene
stage):

.. code::

    P_w = T_wc @ [X_c, Y_c, Z_c, 1]^T

Depth is sampled at the nearest integer pixel with the coordinates clipped
into ``[0, W-1] x [0, H-1]`` so mildly out-of-frame tracks are handled
gracefully rather than crashing.
"""

from __future__ import annotations

import numpy as np


def lift(
    tracks_2d: np.ndarray,
    depth_maps: np.ndarray,
    camera_poses: np.ndarray,
    intrinsics: np.ndarray,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """Back-project 2D tracks to world coordinates.

    Parameters
    ----------
    tracks_2d:
        ``(T, N, 2)`` float, pixel coordinates ``(u, v)``.
    depth_maps:
        ``(T, H, W)`` float, per-pixel metric depth in metres.
    camera_poses:
        ``(T, 4, 4)`` camera-to-world transforms.
    intrinsics:
        ``(3, 3)`` pinhole matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
    frame_size:
        ``(width, height)`` used to clip pixel indices.

    Returns
    -------
    np.ndarray
        ``(T, N, 3)`` float world-frame trajectories.
    """

    t, n, _ = tracks_2d.shape
    if t == 0 or n == 0:
        return np.zeros((t, n, 3), dtype=np.float32)

    width, height = frame_size
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    u = tracks_2d[..., 0]
    v = tracks_2d[..., 1]

    # Nearest-pixel indices, clipped into the frame.
    u_idx = np.clip(np.rint(u).astype(np.int32), 0, width - 1)
    v_idx = np.clip(np.rint(v).astype(np.int32), 0, height - 1)

    # Sample depth per (t, n).
    t_idx = np.arange(t, dtype=np.int32)[:, None]  # (T, 1)
    z = depth_maps[t_idx, v_idx, u_idx].astype(np.float32)  # (T, N)

    # Camera-frame coordinates.
    # Y is flipped so that world-up is +Y, matching scene.py's convention.
    # Without this, tracks live in an inverted-Y frame relative to the point
    # cloud and the trajectory + reconstruction never visually align.
    x_c = (u - cx) * z / fx
    y_c = -(v - cy) * z / fy
    z_c = z

    # Homogeneous points, shape (T, N, 4).
    ones = np.ones_like(z_c)
    p_cam = np.stack([x_c, y_c, z_c, ones], axis=-1)

    # Batched multiply: (T, 4, 4) @ (T, 4, N) -> (T, 4, N)
    p_cam_T = np.transpose(p_cam, (0, 2, 1))  # (T, 4, N)
    p_world_T = np.matmul(camera_poses.astype(np.float32), p_cam_T)
    p_world = np.transpose(p_world_T, (0, 2, 1))  # (T, N, 4)

    return p_world[..., :3].astype(np.float32)
