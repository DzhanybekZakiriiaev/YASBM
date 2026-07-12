"""Camera pose + colored scene point cloud.

Real integration target: **VGGT** (Meta, 2025) or MonST3R for full 4D
reconstruction with recovered camera pose. Until that ships, we use a
practical shortcut:

- Assume a static-ish camera (identity pose per frame).
- Build a coloured point cloud by back-projecting a per-pixel temporal
  *median* RGB plate through the median depth map via a pinhole model.
  When ``exclude_masks`` (per-frame moving-object masks, built by the
  caller from YOLO boxes) are provided, the median for each pixel only
  uses the frames where that pixel is NOT covered by an object — the
  background revealed when the object moves away.

This is not a full 4D reconstruction but it *is* geometrically consistent
with the depth map + trajectories — the point cloud aligns with what
happens in the video, which is all we need for the cinematic viewer.
"""

from __future__ import annotations

import warnings
from typing import TypedDict

import numpy as np


class SceneOutput(TypedDict):
    camera_poses: np.ndarray  # (T, 4, 4) float32
    xyz: np.ndarray  # (N, 3) float32 world-frame points, metres
    rgb: np.ndarray  # (N, 3) uint8 per-vertex color
    # Triangle indices (M, 3) int32 into xyz. Empty ndarray when meshing
    # is disabled or degenerates. Enables rendering the scene as a solid
    # mesh instead of a sparse point cloud.
    faces: np.ndarray
    # Per-frame dynamic (moving-pixel) point clouds. One entry per frame:
    # {"xyz": (K, 3) float32, "rgb": (K, 3) uint8}, K capped per frame.
    # These are the pixels excluded from the static mesh — the browser
    # renders the entry matching the playhead so moving objects scrub
    # through 3D space in sync with the video timeline.
    dynamic: list


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
    max_points: int = 60_000,
    exclude_masks: np.ndarray | None = None,
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
    exclude_masks:
        Optional ``(T, H, W)`` bool — True where a pixel belongs to a
        moving/agent *object* at that frame (built by the caller from
        detector boxes). Excluded pixels are dropped from the background
        RGB median, removed from the static mesh where they dominate,
        and used to sample per-frame dynamic point cutouts.
    """

    t = len(frames)
    if t == 0:
        return SceneOutput(
            camera_poses=np.zeros((0, 4, 4), dtype=np.float32),
            xyz=np.zeros((0, 3), dtype=np.float32),
            rgb=np.zeros((0, 3), dtype=np.uint8),
            faces=np.zeros((0, 3), dtype=np.int32),
            dynamic=[],
        )

    identity = np.broadcast_to(np.eye(4, dtype=np.float32), (t, 4, 4)).copy()

    if depth_maps is None or depth_maps.size == 0:
        # No depth → return a small synthetic cloud so the viewer isn't empty.
        rng = np.random.default_rng(0)
        xyz = rng.random((512, 3), dtype=np.float32) * np.array(
            [2.0, 2.0, 4.0], dtype=np.float32
        ) - np.array([1.0, 1.0, 2.0], dtype=np.float32)
        rgb = np.full((512, 3), 128, dtype=np.uint8)
        return SceneOutput(
            camera_poses=identity,
            xyz=xyz,
            rgb=rgb,
            faces=np.zeros((0, 3), dtype=np.int32),
            dynamic=[],
        )

    # Median depth across frames — robust to moving objects. If a person
    # walks across a static camera, most frames see the wall behind them,
    # so the temporal median picks the wall depth rather than blending
    # foreground + background into an unusable soup.
    depth_stack = depth_maps.astype(np.float32)
    depth = np.median(depth_stack, axis=0)  # (H, W) — median depth per pixel.

    # Per-pixel motion mask. Pixels whose depth varies a lot across frames
    # are moving foreground (person walking, hands, anything dynamic). We
    # drop those from the mesh construction entirely so the reconstructed
    # room is pure static geometry — clean walls / bed / lamp — and the
    # frontend's animated TrackMarkers do the "person walking through" job.
    # Threshold = 15% of the median depth for that pixel, clamped so we
    # don't over-remove noise on distant walls.
    depth_iqr = np.percentile(depth_stack, 75, axis=0) - np.percentile(
        depth_stack, 25, axis=0
    )
    motion_threshold = np.maximum(0.15 * depth, 0.25)  # metres
    static_mask = depth_iqr < motion_threshold  # True where the pixel is stable

    height, width = depth.shape

    # Validate the exclude masks before trusting them: (T, H, W) bool
    # aligned with the depth maps. Anything else is silently ignored so a
    # malformed caller degrades to the old behaviour instead of crashing.
    if exclude_masks is not None:
        exclude_masks = np.asarray(exclude_masks)
        if exclude_masks.shape != (t, height, width):
            exclude_masks = None
        else:
            exclude_masks = exclude_masks.astype(bool, copy=False)

    # Background RGB plate: per-pixel temporal median of RGB. Robust to
    # moving objects the same way the depth median is — a person walking
    # through leaves the wall colour, not their shirt. With exclude_masks
    # the median only uses frames where the pixel is NOT covered by an
    # object; pixels covered in ALL frames fall back to the plain median
    # (the object never moved — it stays part of the scene).
    frame_stack = np.stack(frames, axis=0).astype(np.float32)  # (T, H, W, 3)
    rgb_plate = np.median(frame_stack, axis=0)  # (H, W, 3) float32
    if exclude_masks is not None and exclude_masks.any():
        frame_stack[exclude_masks] = np.nan
        with warnings.catch_warnings():
            # All-NaN pixels (excluded every frame) warn + yield NaN;
            # we substitute the plain median for those below.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            masked_median = np.nanmedian(frame_stack, axis=0)
        rgb_plate = np.where(np.isnan(masked_median), rgb_plate, masked_median)
    del frame_stack  # ~100 MB for 30×480×640 — release eagerly
    rgb_plate = np.clip(rgb_plate, 0.0, 255.0).astype(np.uint8)

    # Pixels occupied by a moving object in a big share of the frames are
    # never trustworthy static geometry, even when their depth barely
    # varies (a mostly-stationary person has low depth IQR but must not be
    # baked into the room mesh as a blob).
    if exclude_masks is not None:
        exclude_any = exclude_masks.mean(axis=0) >= 0.4  # (H, W)
        static_mask = static_mask & ~exclude_any

    # Choose a stride so we end up with ~max_points samples on the actual
    # frame dimensions. Solving `(H*W)/stride^2 <= max_points`.
    stride = max(1, int(np.ceil(np.sqrt(height * width / max_points))))

    intrinsics = _default_intrinsics(width, height)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    ys = np.arange(0, height, stride)
    xs = np.arange(0, width, stride)
    n_rows = len(ys)
    n_cols = len(xs)
    grid_v, grid_u = np.meshgrid(ys, xs, indexing="ij")

    z = depth[grid_v, grid_u].astype(np.float32)  # (n_rows, n_cols)
    x = (grid_u - cx) * z / fx
    y = -(grid_v - cy) * z / fy  # flip Y so up is +Y in world frame
    is_static_grid = static_mask[grid_v, grid_u]  # (n_rows, n_cols) bool

    xyz_grid = np.stack([x, y, z], axis=-1).astype(np.float32)  # (n_rows, n_cols, 3)
    xyz = xyz_grid.reshape(-1, 3)
    rgb = rgb_plate[grid_v, grid_u].reshape(-1, 3).astype(np.uint8)

    faces = _build_edge_filtered_faces(
        xyz_grid, n_rows, n_cols, static_grid=is_static_grid
    )

    dynamic = _dynamic_points_per_frame(
        frames=frames,
        depth_stack=depth_stack,
        static_mask=static_mask,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        max_points_per_frame=2_500,
        exclude_masks=exclude_masks,
    )

    return SceneOutput(
        camera_poses=identity, xyz=xyz, rgb=rgb, faces=faces, dynamic=dynamic
    )


def build_exclude_masks(
    object_reports: list[dict],
    num_frames: int,
    height: int,
    width: int,
    margin: float = 0.08,
) -> np.ndarray | None:
    """Rasterise moving-object boxes from the ``objects`` stage reports
    into per-frame exclusion masks for :func:`scene`.

    An object is "moving" (and therefore excluded from the static room)
    when its verdict is ``agent`` / ``morphing``, or its ballistic audit
    marked it ``eligible`` (free-moving rigid body) or ``self-propelled``.
    ``static`` objects stay — they are part of the room.

    Each box (``boxes_norm``: frame-index string -> [x0, y0, x1, y1]
    normalized 0..1) is expanded by ``margin`` (fraction of box size) on
    all sides, clamped to the frame, to catch silhouette edges.

    Returns ``(num_frames, height, width)`` bool, or ``None`` when no
    pixel was excluded (callers then keep the legacy scene behaviour).
    """

    masks = np.zeros((num_frames, height, width), dtype=bool)
    for report in object_reports:
        ballistic = report.get("ballistic") or {}
        moving = (
            report.get("verdict") in ("agent", "morphing")
            or bool(ballistic.get("eligible"))
            or ballistic.get("reason") == "self-propelled"
        )
        if not moving:
            continue
        for frame_key, box in (report.get("boxes_norm") or {}).items():
            f_idx = int(frame_key)
            if not 0 <= f_idx < num_frames:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            dx = margin * (x1 - x0)
            dy = margin * (y1 - y0)
            c0 = max(0, int(np.floor((x0 - dx) * width)))
            c1 = min(width, int(np.ceil((x1 + dx) * width)))
            r0 = max(0, int(np.floor((y0 - dy) * height)))
            r1 = min(height, int(np.ceil((y1 + dy) * height)))
            if r1 > r0 and c1 > c0:
                masks[f_idx, r0:r1, c0:c1] = True

    return masks if masks.any() else None


def _dynamic_points_per_frame(
    frames: list[np.ndarray],
    depth_stack: np.ndarray,
    static_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_points_per_frame: int = 2_500,
    exclude_masks: np.ndarray | None = None,
) -> list[dict]:
    """Back-project each frame's *moving* pixels through that frame's depth.

    With ``exclude_masks`` (T, H, W) each frame samples from THAT frame's
    object mask — a clean per-frame cutout of the moving objects, colored
    from that frame's RGB. Without it we fall back to the complement of
    the global static mask. Either way sampled at a stride that keeps each
    frame under ``max_points_per_frame`` so the browser payload stays lean
    (~30 frames × 2.5k pts ≈ manageable JSON).
    """

    def _empty() -> dict:
        return {
            "xyz": np.zeros((0, 3), np.float32),
            "rgb": np.zeros((0, 3), np.uint8),
        }

    if exclude_masks is not None:
        out_masked: list[dict] = []
        for f_idx, frame in enumerate(frames):
            mask = exclude_masks[f_idx]
            n_pixels = int(mask.sum())
            if n_pixels == 0:
                out_masked.append(_empty())
                continue
            vs, us = np.nonzero(mask)
            stride = max(1, int(np.ceil(n_pixels / max_points_per_frame)))
            vs, us = vs[::stride], us[::stride]
            z = depth_stack[f_idx, vs, us].astype(np.float32)
            x = (us - cx) * z / fx
            y = -(vs - cy) * z / fy
            out_masked.append(
                {
                    "xyz": np.stack([x, y, z], axis=1).astype(np.float32),
                    "rgb": frame[vs, us].astype(np.uint8),
                }
            )
        return out_masked

    moving_mask = ~static_mask
    n_moving = int(moving_mask.sum())
    out: list[dict] = []

    if n_moving == 0:
        return [
            {"xyz": np.zeros((0, 3), np.float32), "rgb": np.zeros((0, 3), np.uint8)}
            for _ in frames
        ]

    vs, us = np.nonzero(moving_mask)
    stride = max(1, int(np.ceil(n_moving / max_points_per_frame)))
    vs, us = vs[::stride], us[::stride]

    for f_idx, frame in enumerate(frames):
        z = depth_stack[f_idx, vs, us].astype(np.float32)
        x = (us - cx) * z / fx
        y = -(vs - cy) * z / fy
        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        rgb = frame[vs, us].astype(np.uint8)
        out.append({"xyz": xyz, "rgb": rgb})

    return out


def _build_edge_filtered_faces(
    xyz_grid: np.ndarray,
    n_rows: int,
    n_cols: int,
    static_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Delaunay-in-a-grid: two triangles per cell, drop any triangle whose
    3D edge lengths exceed ~3× the median cell edge. That kills the huge
    stretched triangles that would otherwise bridge foreground objects and
    the wall behind them — the "cardboard-cutout depth-image" look.

    If ``static_grid`` is provided (bool array matching (n_rows, n_cols)),
    also drop any triangle whose vertices include a moving-object pixel.
    That keeps the mesh pure static room — moving people are not baked
    into flat cutouts on the wall.
    """

    if n_rows < 2 or n_cols < 2:
        return np.zeros((0, 3), dtype=np.int32)

    # Vertex index into the flat xyz array: (r, c) -> r * n_cols + c.
    r = np.arange(n_rows - 1)
    c = np.arange(n_cols - 1)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    top_left = (rr * n_cols + cc).reshape(-1)
    top_right = top_left + 1
    bot_left = top_left + n_cols
    bot_right = bot_left + 1

    # Two triangles per cell.
    tri_a = np.stack([top_left, bot_right, top_right], axis=1)
    tri_b = np.stack([top_left, bot_left, bot_right], axis=1)
    all_faces = np.concatenate([tri_a, tri_b], axis=0)

    # Filter by edge length: compute max edge per triangle in 3D.
    xyz_flat = xyz_grid.reshape(-1, 3)
    p0 = xyz_flat[all_faces[:, 0]]
    p1 = xyz_flat[all_faces[:, 1]]
    p2 = xyz_flat[all_faces[:, 2]]
    e01 = np.linalg.norm(p1 - p0, axis=1)
    e12 = np.linalg.norm(p2 - p1, axis=1)
    e20 = np.linalg.norm(p0 - p2, axis=1)
    max_edge = np.maximum(np.maximum(e01, e12), e20)

    median_edge = float(np.median(max_edge)) if max_edge.size else 0.0
    threshold = max(median_edge * 3.0, 0.01)  # metres
    keep = max_edge < threshold

    if static_grid is not None:
        static_flat = static_grid.reshape(-1)
        # Keep a triangle only if ALL three of its vertices are static.
        vertex_static = (
            static_flat[all_faces[:, 0]]
            & static_flat[all_faces[:, 1]]
            & static_flat[all_faces[:, 2]]
        )
        keep = keep & vertex_static

    return all_faces[keep].astype(np.int32)
