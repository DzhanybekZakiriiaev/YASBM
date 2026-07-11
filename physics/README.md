# kepler-physics

Standalone Newtonian projectile-with-linear-drag fit for the KEPLER
AI-video plausibility auditor. Given a rigid object's 3D trajectory over
time, it fits `a = g - drag * v` and reports where and how much the
observed motion diverges from Newtonian dynamics.

Real footage tends to fit cleanly. AI-generated footage often leaves a
measurable physics residual.

## Install / test

From `physics/`:

```powershell
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
uv sync
uv run pytest -v
```

## Quick example

```python
import numpy as np
from kepler_physics import fit

ts = np.linspace(0.0, 1.0, 30)
positions = ...  # shape (T, 3) — world-frame trajectory in meters
result = fit(positions, ts)

print(result.v0, result.g, result.drag)
print("peak_sigma:", result.peak_sigma)  # >~ 5 suggests physics violation
print("rmse:", result.rmse)
```

`result.residuals` (shape `(T, 3)`) and `result.residual_norm`
(shape `(T,)`) are the per-frame physics-violation signal you can plot
alongside the trajectory.

## Dependencies

`numpy >= 2.1`, `scipy >= 1.14`. No ML / no torch.
