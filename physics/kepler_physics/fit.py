"""Levenberg-Marquardt fit of Newtonian projectile-with-drag dynamics.

Given an observed 3D trajectory ``positions`` sampled at ``timestamps``,
``fit`` recovers the initial velocity ``v0``, the gravity vector ``g`` and
the linear-drag coefficient ``drag`` that best reproduce the observations.

The returned :class:`FitResult` contains per-frame residuals (observed
minus predicted, signed 3-vectors) and their magnitudes, together with a
summary ``peak_sigma``: the largest residual magnitude divided by an
empirical noise floor estimated from the trajectory's own
second-differences. Real footage tends to produce ``peak_sigma`` values
near unity; AI-generated footage often spikes far above that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from kepler_physics.model import integrate


@dataclass
class FitResult:
    v0: np.ndarray  # (3,)
    g: np.ndarray  # (3,)
    drag: float
    residuals: np.ndarray  # (T, 3) signed: observed - predicted
    residual_norm: np.ndarray  # (T,) magnitude of residuals
    peak_sigma: float
    rmse: float
    converged: bool


def _pack(v0: np.ndarray, g: np.ndarray, drag: float) -> np.ndarray:
    return np.concatenate([v0, g, [drag]])


def _unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    v0 = np.asarray(params[0:3], dtype=float)
    g = np.asarray(params[3:6], dtype=float)
    drag = float(params[6])
    return v0, g, drag


def fit(
    positions: np.ndarray,
    timestamps: np.ndarray,
    fix_gravity_mag: bool = False,
) -> FitResult:
    """Fit Newtonian projectile-with-drag dynamics to ``positions``.

    Parameters
    ----------
    positions : (T, 3) observed world-frame positions.
    timestamps : (T,) sample times in seconds.
    fix_gravity_mag : reserved for a future constrained variant. Currently
        the gravity vector is free.

    Returns
    -------
    FitResult with per-frame residuals and summary statistics.

    Notes
    -----
    The dynamics x(t) = x0 + integral(v(t) dt) with a(t) = g - drag * v(t)
    is genuinely underdetermined for near-static trajectories: any triple
    (v0, g, drag) where drag * dt >> 1 damps the motion to near-x0 fits
    noisy static data equally well. To avoid landing in that degenerate
    valley, we solve the drag-free (linear-in-parameters) sub-problem
    first via closed-form least squares, then only accept the nonlinear
    LM refinement if it meaningfully reduces the residual RMSE. This
    functions as a soft prior peaked at drag = 0 -- the simpler model
    wins unless the data really demands drag.
    """
    positions = np.asarray(positions, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must have shape (T, 3), got {positions.shape}")
    if timestamps.ndim != 1 or timestamps.shape[0] != positions.shape[0]:
        raise ValueError(
            f"timestamps must have shape (T,) matching positions; got "
            f"{timestamps.shape} vs {positions.shape}"
        )
    T = positions.shape[0]
    if T < 4:
        raise ValueError(f"need at least 4 samples to fit, got {T}")

    x0 = positions[0].copy()

    # `fix_gravity_mag` is accepted for API compatibility but not yet
    # constrained; the LM solver operates on all seven parameters.
    _ = fix_gravity_mag

    # Stage 1: closed-form linear fit for the drag-free model.
    # x(t) = x0 + v0 * dt + 0.5 * g * dt^2 is linear in (v0, g); solve
    # independently per axis via ordinary least squares.
    dt_all = timestamps - timestamps[0]
    delta = positions - x0[None, :]  # (T, 3)
    A = np.stack([dt_all, 0.5 * dt_all**2], axis=1)  # (T, 2)
    coeffs, *_ = np.linalg.lstsq(A, delta, rcond=None)  # (2, 3)
    v0_lin = coeffs[0]
    g_lin = coeffs[1]
    predicted_lin = (
        x0[None, :] + v0_lin[None, :] * dt_all[:, None] + 0.5 * g_lin[None, :] * dt_all[:, None] ** 2
    )
    rmse_lin = float(np.sqrt(np.mean((positions - predicted_lin) ** 2)))

    # Stage 2: nonlinear LM refinement seeded from the linear fit.
    def residual_fn(params: np.ndarray) -> np.ndarray:
        v0, g, drag = _unpack(params)
        predicted = integrate(x0, v0, g, drag, timestamps)
        return (predicted - positions).ravel()

    params0 = _pack(v0_lin, g_lin, 0.0)
    result = least_squares(residual_fn, params0, method="lm")

    v0_nl, g_nl, drag_nl = _unpack(result.x)
    predicted_nl = integrate(x0, v0_nl, g_nl, drag_nl, timestamps)
    rmse_nl = float(np.sqrt(np.mean((positions - predicted_nl) ** 2)))

    # Stage 3: model selection. The drag term earns its place only if it
    # cuts the RMSE by at least ~30%. The threshold 0.7 encodes a mild
    # simplicity prior: for noisy static data both fits have essentially
    # the same RMSE and we keep drag = 0, avoiding the degeneracy where
    # LM inflates (|v0|, |g|, drag) together along a low-cost ridge.
    threshold = 0.7
    if rmse_nl < threshold * rmse_lin:
        v0_out = v0_nl
        g_out = g_nl
        drag_out = drag_nl
        predicted = predicted_nl
        converged = bool(result.success)
    else:
        v0_out = v0_lin
        g_out = g_lin
        drag_out = 0.0
        predicted = predicted_lin
        converged = True  # closed-form solution always succeeds

    residuals = positions - predicted  # signed observed - predicted
    residual_norm = np.linalg.norm(residuals, axis=1)

    # Noise floor: median magnitude of the second-difference of positions
    # over interior samples. Second differences of a smooth trajectory
    # are ~ a * dt^2, so this is a robust proxy for measurement noise.
    if T >= 3:
        second_diff = positions[2:] - 2 * positions[1:-1] + positions[:-2]
        second_diff_norm = np.linalg.norm(second_diff, axis=1)
        noise_floor = float(np.median(second_diff_norm))
    else:
        noise_floor = 0.0

    peak_sigma = float(np.max(residual_norm) / max(noise_floor, 1e-6))
    rmse = float(np.sqrt(np.mean(residual_norm**2)))

    return FitResult(
        v0=v0_out,
        g=g_out,
        drag=drag_out,
        residuals=residuals,
        residual_norm=residual_norm,
        peak_sigma=peak_sigma,
        rmse=rmse,
        converged=converged,
    )
