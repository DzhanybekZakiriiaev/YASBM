"""KEPLER physics-fit module.

Fits a Newtonian projectile-with-linear-drag model to a 3D rigid-object
trajectory and reports per-frame residuals plus a peak-sigma summary that
quantifies where and how much the motion violated physics.

Real footage yields low residuals; AI-generated footage tends to leave a
measurable physics residual.
"""

from kepler_physics.fit import FitResult, fit

__all__ = ["FitResult", "fit"]
