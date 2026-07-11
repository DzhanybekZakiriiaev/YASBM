"""Physics-fit stage — delegates to the standalone ``kepler_physics`` module.

Newton's laws are applied per-track: ``kepler_physics.fit`` recovers
initial velocity + gravity + linear drag from each 3D trajectory, then
reports residual magnitude per frame and a summary ``peak_sigma``
(largest residual normalised by an empirical noise floor).

The sibling ``../../../physics/`` project is added to ``sys.path`` so
we can import it without configuring a formal monorepo. For deployment,
replace this with a proper dependency in ``pyproject.toml`` via
``[tool.uv.sources] kepler-physics = { path = "../physics", editable = true }``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# In local dev, kepler_physics lives at ../../../physics. In the Modal
# container it's copied to /root/kepler_physics by add_local_dir. Try both.
_CANDIDATE_ROOTS = [
    Path(__file__).resolve().parents[3] / "physics",
    Path("/root"),
]
for _root in _CANDIDATE_ROOTS:
    if (_root / "kepler_physics").exists() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from kepler_physics import fit as fit_trajectory  # noqa: E402

# Minimum samples the ``kepler_physics.fit`` LM solver needs.
_MIN_SAMPLES = 4


def physics(tracks_3d: np.ndarray, timestamps: np.ndarray) -> dict:
    """Fit Newtonian dynamics to each 3D track; return residuals + verdict.

    Parameters
    ----------
    tracks_3d:
        ``(T, N, 3)`` world-frame trajectories in metres.
    timestamps:
        ``(T,)`` sample times in seconds.

    Returns
    -------
    dict
        - ``residuals``: ``list[np.ndarray (T,)]`` of residual magnitudes,
          one entry per track (kept for backward compat with earlier
          serialisation code).
        - ``per_frame_max``: ``np.ndarray (T,)`` — max residual magnitude
          across tracks at each frame. Drives the response's ``delta_m``.
        - ``per_frame_sigma``: ``np.ndarray (T,)`` — per-frame σ.
        - ``peak_sigma``: ``float`` — max σ across all tracks and frames.
        - ``noise_floor``: ``float`` — median second-difference magnitude.
        - ``track_fits``: ``list`` of ``FitResult | None`` per track.
    """

    if tracks_3d.ndim != 3 or tracks_3d.shape[2] != 3:
        raise ValueError(f"tracks_3d must be (T, N, 3), got {tracks_3d.shape}")

    T = int(tracks_3d.shape[0])
    N = int(tracks_3d.shape[1])

    per_track_residuals: list[np.ndarray] = []
    track_fits: list = []
    peak_sigma = 0.0
    noise_floor_accum: list[float] = []

    if T < _MIN_SAMPLES:
        # Not enough samples to fit anything.
        zeros = np.zeros(T, dtype=np.float32)
        return {
            "residuals": [zeros.copy() for _ in range(N)],
            "per_frame_max": zeros.copy(),
            "per_frame_sigma": zeros.copy(),
            "peak_sigma": 0.0,
            "noise_floor": 0.0,
            "track_fits": [None] * N,
        }

    for track_id in range(N):
        positions = tracks_3d[:, track_id, :].astype(np.float64)
        try:
            result = fit_trajectory(positions, timestamps.astype(np.float64))
            track_fits.append(result)
            per_track_residuals.append(result.residual_norm.astype(np.float32))
            peak_sigma = max(peak_sigma, float(result.peak_sigma))
            # Recover the noise floor this fit was normalised against so we can
            # aggregate across tracks. Guard divide-by-zero.
            max_norm = float(np.max(result.residual_norm))
            if result.peak_sigma > 1e-9:
                noise_floor_accum.append(max_norm / result.peak_sigma)
        except (ValueError, np.linalg.LinAlgError):
            track_fits.append(None)
            per_track_residuals.append(np.zeros(T, dtype=np.float32))

    stacked = np.stack(per_track_residuals, axis=0)  # (N, T)
    per_frame_max = stacked.max(axis=0)  # (T,)

    # Aggregate noise floor across tracks. Use median for robustness against
    # a single degenerate track dragging the estimate.
    if noise_floor_accum:
        noise_floor = float(np.median(noise_floor_accum))
    else:
        noise_floor = 0.0

    per_frame_sigma = per_frame_max / max(noise_floor, 1e-6)

    return {
        "residuals": per_track_residuals,
        "per_frame_max": per_frame_max.astype(np.float32),
        "per_frame_sigma": per_frame_sigma.astype(np.float32),
        "peak_sigma": float(peak_sigma),
        "noise_floor": noise_floor,
        "track_fits": track_fits,
    }
