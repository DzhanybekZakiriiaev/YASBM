"""Tests for kepler_physics.fit — covering perfect, noisy, adversarial, and
static-object trajectories."""

from __future__ import annotations

import numpy as np

from kepler_physics.fit import fit
from kepler_physics.model import integrate


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def test_perfect_projectile_no_drag() -> None:
    true_v0 = np.array([2.0, 5.0, -1.0])
    true_g = np.array([0.0, -9.81, 0.0])
    x0 = np.array([0.0, 0.0, 0.0])
    ts = np.linspace(0.0, 1.0, 30)

    positions = integrate(x0, true_v0, true_g, 0.0, ts)
    result = fit(positions, ts)

    v0_err = _rel(result.v0, true_v0)
    g_err = _rel(result.g, true_g)

    print(
        f"[test_perfect_projectile_no_drag] peak_sigma={result.peak_sigma:.6g} "
        f"rmse={result.rmse:.3e} v0_err={v0_err:.3e} g_err={g_err:.3e}"
    )

    assert result.converged
    assert v0_err < 0.01
    assert g_err < 0.02
    assert result.peak_sigma < 1.0


def test_projectile_with_noise() -> None:
    rng = np.random.default_rng(0)
    true_v0 = np.array([1.5, 6.0, 0.5])
    true_g = np.array([0.0, -9.81, 0.0])
    x0 = np.array([0.0, 1.0, 0.0])
    ts = np.linspace(0.0, 1.0, 30)

    positions = integrate(x0, true_v0, true_g, 0.0, ts)
    positions = positions + rng.normal(scale=0.005, size=positions.shape)

    result = fit(positions, ts)

    print(
        f"[test_projectile_with_noise] peak_sigma={result.peak_sigma:.6g} "
        f"rmse={result.rmse:.3e}"
    )

    assert result.converged
    assert result.peak_sigma < 5.0
    assert result.rmse < 0.02


def test_adversarial_teleport() -> None:
    rng = np.random.default_rng(1)
    true_v0 = np.array([3.0, 4.0, 0.0])
    true_g = np.array([0.0, -9.81, 0.0])
    x0 = np.array([0.0, 0.0, 0.0])
    ts = np.linspace(0.0, 1.0, 30)

    positions = integrate(x0, true_v0, true_g, 0.0, ts)
    # A little noise so the noise floor is realistic (not near zero).
    positions = positions + rng.normal(scale=0.003, size=positions.shape)

    teleport_idx = ts.size // 2
    positions[teleport_idx:, 0] += 0.5  # lateral teleport in x

    result = fit(positions, ts)

    print(
        f"[test_adversarial_teleport] peak_sigma={result.peak_sigma:.6g} "
        f"rmse={result.rmse:.3e}"
    )

    assert result.peak_sigma > 5.0


def test_static_object() -> None:
    rng = np.random.default_rng(2)
    ts = np.linspace(0.0, 1.0, 30)
    positions = np.tile(np.array([1.0, 2.0, 3.0]), (ts.size, 1))
    positions = positions + rng.normal(scale=0.001, size=positions.shape)

    result = fit(positions, ts)

    v0_mag = float(np.linalg.norm(result.v0))
    print(
        f"[test_static_object] peak_sigma={result.peak_sigma:.6g} "
        f"rmse={result.rmse:.3e} |v0|={v0_mag:.3e}"
    )

    assert v0_mag < 0.1
    assert result.peak_sigma < 3.0
