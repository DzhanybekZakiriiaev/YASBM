"""CoTracker3 offline dense point tracking.

Loads the offline CoTracker3 checkpoint via ``torch.hub`` and runs it over
the full clip with an ``NxN`` grid seed. Default grid_size=8 gives a 64-
point track set, which is what the KEPLER lift + fit pipeline expects.

When torch (or CoTracker3's dependencies) can't be imported — the local
dev environment without heavy ML deps — this module returns synthetic
straight-line tracks so the rest of the pipeline still runs end-to-end.

Docs referenced:
- https://github.com/facebookresearch/co-tracker
- https://github.com/facebookresearch/co-tracker/blob/main/README.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

_HANDLE: dict[str, Any] | None = None


def _try_load() -> dict[str, Any] | None:
    """Attempt to load CoTracker3. Return None on any failure."""

    if not _HAS_TORCH:
        return None
    try:
        return load_tracker()
    except Exception:  # pragma: no cover - defensive against hub errors
        return None


def load_tracker() -> dict[str, Any]:
    """Load CoTracker3 offline from ``facebookresearch/co-tracker`` hub."""

    import torch  # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    tracker = tracker.to(device).eval()
    return {"tracker": tracker, "device": device}


def prefetch() -> None:
    """Warm-load and cache the tracker at module level.

    Called from Modal's ``@modal.enter()`` so the first ``analyze`` request
    pays only inference cost, not the ~10s hub download + load.
    """

    global _HANDLE
    if _HANDLE is None:
        _HANDLE = _try_load()


def _stub_tracks(frames: list[np.ndarray], grid_size: int) -> np.ndarray:
    """Synthetic left-to-right sweep so downstream stages still run."""

    n_points = grid_size * grid_size
    t = len(frames)
    if t == 0:
        return np.zeros((0, n_points, 2), dtype=np.float32)
    height, width = frames[0].shape[:2]
    xs = np.linspace(0, width - 1, grid_size, dtype=np.float32)
    ys = np.linspace(0, height - 1, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    base = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)  # (N, 2)
    sweep = np.linspace(-0.1 * width, 0.1 * width, t, dtype=np.float32)
    tracks = np.stack(
        [base + np.array([[dx, 0.0]], dtype=np.float32) for dx in sweep],
        axis=0,
    )
    return tracks.reshape(t, n_points, 2)


def track(
    frames: list[np.ndarray],
    handle: dict[str, Any] | None = None,
    grid_size: int = 8,
) -> np.ndarray:
    """Run CoTracker3 offline (or a stub fallback).

    Parameters
    ----------
    frames:
        List of RGB frames, ``(H, W, 3)`` ``uint8``.
    handle:
        Preloaded handle from :func:`load_tracker`, or ``None`` to lazy-load.
        If loading fails (no torch installed) the function falls back to a
        synthetic straight-line track set.
    grid_size:
        Side length of the seed grid. ``grid_size=8`` -> 64 tracked points.
    """

    n_points = grid_size * grid_size
    if not frames:
        return np.zeros((0, n_points, 2), dtype=np.float32)

    # Preference order: explicit handle → module cache → lazy load → stub.
    if handle is None:
        handle = _HANDLE
    if handle is None:
        handle = _try_load()
    if handle is None:
        return _stub_tracks(frames, grid_size)

    import torch

    device = handle["device"]
    tracker = handle["tracker"]

    arr = np.stack(frames, axis=0).astype(np.float32)
    video = (
        torch.from_numpy(arr)
        .permute(0, 3, 1, 2)  # (T, 3, H, W)
        .unsqueeze(0)  # (1, T, 3, H, W)
        .to(device)
    )

    with torch.inference_mode():
        pred_tracks, _pred_vis = tracker(video, grid_size=grid_size)

    return pred_tracks[0].detach().cpu().numpy().astype(np.float32)
