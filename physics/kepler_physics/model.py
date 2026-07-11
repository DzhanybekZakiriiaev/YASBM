"""Newtonian rigid-body projectile motion with linear drag.

Dynamics: a(t) = g - drag * v(t), where
  g is the gravity vector (world-frame, e.g. (0, -9.81, 0))
  drag is a scalar aerodynamic coefficient with units 1/s.

Real footage produces trajectories consistent with these dynamics.
AI-generated footage often does not.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def integrate(
    x0: np.ndarray,
    v0: np.ndarray,
    g: np.ndarray,
    drag: float,
    ts: np.ndarray,
) -> np.ndarray:
    """Integrate rigid-body dynamics a = g - drag * v.

    Parameters
    ----------
    x0 : (3,) initial position (world frame, meters).
    v0 : (3,) initial velocity (m/s).
    g  : (3,) gravity vector (m/s^2).
    drag : scalar linear-drag coefficient (1/s). If exactly zero, uses the
        closed-form projectile solution x(t) = x0 + v0 t + 0.5 g t^2.
    ts : (T,) sample times, monotonically increasing. ts[0] is treated as
        the start time; positions returned are evaluated at each ts[i].

    Returns
    -------
    positions : (T, 3) array of world-frame positions at each ts[i].
    """
    x0 = np.asarray(x0, dtype=float).reshape(3)
    v0 = np.asarray(v0, dtype=float).reshape(3)
    g = np.asarray(g, dtype=float).reshape(3)
    ts = np.asarray(ts, dtype=float).reshape(-1)

    if ts.size == 0:
        return np.zeros((0, 3), dtype=float)

    if drag == 0.0:
        # Closed form: x(t) = x0 + v0 * dt + 0.5 * g * dt^2
        dt = (ts - ts[0]).reshape(-1, 1)  # (T, 1)
        return x0[None, :] + v0[None, :] * dt + 0.5 * g[None, :] * dt**2

    # General case: RK45 on state y = [x (3), v (3)]
    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        v = y[3:]
        dx = v
        dv = g - drag * v
        return np.concatenate([dx, dv])

    y0 = np.concatenate([x0, v0])
    t0 = float(ts[0])
    t_end = float(ts[-1])
    # Guard for the degenerate case where all ts are identical.
    if t_end == t0:
        return np.tile(x0, (ts.size, 1))

    sol = solve_ivp(
        rhs,
        (t0, t_end),
        y0,
        method="RK45",
        dense_output=True,
        t_eval=ts,
        rtol=1e-8,
        atol=1e-10,
    )
    # sol.y has shape (6, T); positions are the first three rows.
    return sol.y[:3, :].T
