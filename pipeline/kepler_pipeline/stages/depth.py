"""Depth Anything V2 (Small) per-frame monocular depth.

We run the HuggingFace ``depth-estimation`` pipeline with the checkpoint
``depth-anything/Depth-Anything-V2-Small-hf``. The raw ``predicted_depth``
tensor from the relative-depth head is DISPARITY-like (larger = closer),
so we invert it and then rescale the whole clip so the first-frame median
depth is ~5 metres. This gives the lift stage a plausible metric scale
even though Depth Anything V2 does not natively output metres.

Temporal smoothing (Video Depth Anything) is a follow-up: this frame-wise
version is jitter-sensitive but sufficient to prove out the pipeline.

Falls back to a constant depth stub when transformers/torch are not
importable (local dev without heavy ML deps).

Docs referenced:
- https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2
- https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch  # noqa: F401
    from transformers import pipeline as _hf_pipeline  # noqa: F401

    _HAS_TF = True
except ImportError:  # pragma: no cover
    _HAS_TF = False

_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_HANDLE: dict[str, Any] | None = None


def _try_load() -> dict[str, Any] | None:
    if not _HAS_TF:
        return None
    try:
        return load_depth()
    except Exception:  # pragma: no cover - defensive against HF download errors
        return None


def load_depth() -> dict[str, Any]:
    """Load the Depth Anything V2 Small pipeline."""

    import torch
    from transformers import pipeline as hf_pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline(
        task="depth-estimation",
        model=_DEPTH_MODEL,
        device=device,
    )
    return {"pipe": pipe, "device": device}


def prefetch() -> None:
    """Warm-load and cache the depth pipeline module-level."""

    global _HANDLE
    if _HANDLE is None:
        _HANDLE = _try_load()


def _stub_depth(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros((0, 0, 0), dtype=np.float32)
    h, w = frames[0].shape[:2]
    return np.full((len(frames), h, w), 5.0, dtype=np.float32)


def depth(
    frames: list[np.ndarray],
    handle: dict[str, Any] | None = None,
    target_median_m: float = 5.0,
) -> np.ndarray:
    """Return metric-ish depth stack shaped ``(T, H, W)`` float32 in metres.

    Parameters
    ----------
    frames:
        List of RGB frames ``(H, W, 3)`` ``uint8``.
    handle:
        Preloaded handle from :func:`load_depth`, or ``None`` to lazy-load.
        Falls back to a constant-Z stub if loading fails.
    target_median_m:
        Global scale set so ``median(depth[0]) == target_median_m`` metres.
    """

    t = len(frames)
    if t == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)

    if handle is None:
        handle = _HANDLE
    if handle is None:
        handle = _try_load()
    if handle is None:
        return _stub_depth(frames)

    from PIL import Image

    h, w = frames[0].shape[:2]
    pil_frames = [Image.fromarray(f) for f in frames]

    outputs = handle["pipe"](pil_frames)
    if isinstance(outputs, dict):
        outputs = [outputs]

    raw_maps: list[np.ndarray] = []
    for out in outputs:
        pred = out["predicted_depth"]
        arr = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape != (h, w):
            import cv2

            arr = cv2.resize(
                arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
        raw_maps.append(arr.astype(np.float32))

    disparity = np.stack(raw_maps, axis=0)
    disparity = np.clip(disparity, 1e-3, None)
    inv_depth = 1.0 / disparity

    med0 = float(np.median(inv_depth[0]))
    scale = target_median_m / max(med0, 1e-9)
    depth_m = inv_depth * scale
    depth_m = np.clip(depth_m, 0.1, 100.0)
    return depth_m.astype(np.float32)
