"""3D prop placement for detected objects.

For every object report produced by the ``objects`` stage this computes a
world-space placement (position + approximate bounding-box scale) so the
frontend can drop a proxy 3D model into the reconstructed scene:

- **Representative frame** — the frame index (among frames where the
  object has a box) closest to the median of its present frames; a stable
  middle-of-life view of the object.
- **Depth** — median of that frame's depth map inside the (clipped) box.
- **Back-projection** — the same pinhole convention as ``scene.py`` /
  ``lift.py``: ``fx = fy = frame_width``, ``cx = W/2``, ``cy = H/2``,
  world X right, Y up (v-axis flipped), Z depth into the scene.
- **Crop** — the box region (5% margin) of the representative frame is
  saved as a PNG under ``out_dir/crops/{object_id}.png`` so a later
  ``/hero`` call can feed it to an image-to-3D service.
- **Poly Pizza** — optionally resolves a real GLB for the COCO label via
  the Poly Pizza search API (needs ``POLYPIZZA_API_KEY`` in the env).
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

_POLYPIZZA_SEARCH = "https://api.poly.pizza/v1.1/search/{query}?Limit=1"
_POLYPIZZA_TIMEOUT_S = 5.0

# label -> GLB url (or None when the lookup failed / found nothing).
# Module-level so repeated objects of the same class cost one request.
_GLB_CACHE: dict[str, str | None] = {}

_CROP_MARGIN = 0.05  # fraction of box size added on each side


def _extract_download_url(payload: object) -> str | None:
    """Defensively pull a GLB download URL out of a Poly Pizza response.

    The documented shape is ``{"results": [{"Download": "<glb url>", ...}]}``
    but we tolerate casing/nesting variants and return None on anything
    surprising instead of raising.
    """

    if not isinstance(payload, dict):
        return None
    results = payload.get("results")
    if results is None:
        results = payload.get("Results")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    if not isinstance(first, dict):
        return None
    for key in ("Download", "download", "DownloadUrl", "download_url", "glb", "Glb"):
        value = first.get(key)
        if isinstance(value, str) and value.startswith("http"):
            return value
        if isinstance(value, dict):
            for sub_key in ("url", "URL", "glb", "Glb"):
                sub = value.get(sub_key)
                if isinstance(sub, str) and sub.startswith("http"):
                    return sub
    return None


def resolve_glb(label: str) -> str | None:
    """Resolve a Poly Pizza GLB URL for ``label``; None when unavailable.

    Requires ``POLYPIZZA_API_KEY`` in the environment. Results (including
    misses) are cached per label for the life of the process. Any network
    or parse problem degrades to None — props never fail the pipeline.
    """

    api_key = os.environ.get("POLYPIZZA_API_KEY")
    if not api_key:
        return None
    if label in _GLB_CACHE:
        return _GLB_CACHE[label]

    url = _POLYPIZZA_SEARCH.format(query=urllib.parse.quote(label))
    glb_url: str | None = None
    try:
        req = urllib.request.Request(url, headers={"x-auth-token": api_key})
        with urllib.request.urlopen(req, timeout=_POLYPIZZA_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        glb_url = _extract_download_url(payload)
    except Exception:  # noqa: BLE001 — any failure means "no model", never raise
        glb_url = None

    _GLB_CACHE[label] = glb_url
    return glb_url


def _representative_frame(boxes_norm: dict) -> int | None:
    """Frame index closest to the median of the object's present frames."""

    try:
        present = sorted(int(k) for k in boxes_norm.keys())
    except (TypeError, ValueError):
        return None
    if not present:
        return None
    median = float(np.median(present))
    return min(present, key=lambda f: abs(f - median))


def _save_crop(
    frame: np.ndarray,
    box_px: tuple[float, float, float, float],
    out_dir: Path,
    object_id: int,
) -> bool:
    """Save the box region (with margin, clipped) as a PNG. True on success."""

    height, width = frame.shape[:2]
    x0, y0, x1, y1 = box_px
    mx = _CROP_MARGIN * (x1 - x0)
    my = _CROP_MARGIN * (y1 - y0)
    c0 = max(0, int(np.floor(x0 - mx)))
    c1 = min(width, int(np.ceil(x1 + mx)))
    r0 = max(0, int(np.floor(y0 - my)))
    r1 = min(height, int(np.ceil(y1 + my)))
    if r1 <= r0 or c1 <= c0:
        return False

    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_rgb = frame[r0:r1, c0:c1]
    # Frames are RGB throughout the pipeline; cv2.imwrite expects BGR.
    crop_bgr = cv2.cvtColor(np.ascontiguousarray(crop_rgb), cv2.COLOR_RGB2BGR)
    return bool(cv2.imwrite(str(crop_dir / f"{object_id}.png"), crop_bgr))


def _median_depth_in_box(
    depth_maps: np.ndarray | None,
    frame_idx: int,
    box_px: tuple[float, float, float, float],
    width: int,
    height: int,
) -> float | None:
    """Median depth inside the (clipped) box; None when nothing usable."""

    if depth_maps is None:
        return None
    depth_maps = np.asarray(depth_maps)
    if depth_maps.ndim != 3 or depth_maps.shape[0] == 0:
        return None
    frame_idx = int(np.clip(frame_idx, 0, depth_maps.shape[0] - 1))
    depth = depth_maps[frame_idx]

    x0, y0, x1, y1 = box_px
    c0 = max(0, int(np.floor(x0)))
    c1 = min(width, int(np.ceil(x1)))
    r0 = max(0, int(np.floor(y0)))
    r1 = min(height, int(np.ceil(y1)))

    region = depth[r0:r1, c0:c1] if (r1 > r0 and c1 > c0) else depth[0:0, 0:0]
    values = region[np.isfinite(region)]
    if values.size == 0:
        # Empty / degenerate box — fall back to the whole-frame median.
        values = depth[np.isfinite(depth)]
        if values.size == 0:
            return None
    return float(np.median(values))


def props(
    object_reports: list[dict],
    depth_maps: np.ndarray | None,
    frames: list[np.ndarray],
    frame_size: tuple[int, int],
    out_dir: str | Path | None = None,
    request_id: str | None = None,
) -> list[dict]:
    """Compute a 3D placement (and optional crop + GLB) per object.

    Parameters
    ----------
    object_reports:
        Reports from the ``objects`` stage (``object_id``, ``label``,
        ``boxes_norm``: frame-index string -> [x0, y0, x1, y1] in [0, 1]).
    depth_maps:
        ``(T, H, W)`` per-frame depth in metres (or None).
    frames:
        RGB uint8 frames, aligned with ``depth_maps``.
    frame_size:
        ``(width, height)`` in pixels.
    out_dir:
        When given, object crops are saved to ``out_dir/crops/{id}.png``.
    request_id:
        Used to build the relative ``crop_url``
        (``/artifacts/{request_id}/crops/{id}.png``).
    """

    width, height = int(frame_size[0]), int(frame_size[1])
    if width <= 0 or height <= 0:
        return []
    fx = fy = float(width)
    cx = width / 2.0
    cy = height / 2.0

    out_path = Path(out_dir) if out_dir is not None else None
    placements: list[dict] = []

    for report in object_reports:
        object_id = report.get("object_id")
        label = report.get("label", "")
        boxes_norm = report.get("boxes_norm") or {}
        rep_frame = _representative_frame(boxes_norm)
        if object_id is None or rep_frame is None:
            continue

        box_norm = boxes_norm.get(str(rep_frame), boxes_norm.get(rep_frame))
        if box_norm is None or len(box_norm) != 4:
            continue
        x0 = float(box_norm[0]) * width
        y0 = float(box_norm[1]) * height
        x1 = float(box_norm[2]) * width
        y1 = float(box_norm[3]) * height
        box_px = (x0, y0, x1, y1)

        z = _median_depth_in_box(depth_maps, rep_frame, box_px, width, height)
        if z is None or not np.isfinite(z) or z <= 0:
            z = 1.0  # degenerate depth — park the prop 1 m out rather than drop it

        u = (x0 + x1) / 2.0
        v = (y0 + y1) / 2.0
        pos_x = (u - cx) * z / fx
        pos_y = -(v - cy) * z / fy
        pos_z = z

        sx = max(0.0, (x1 - x0)) * z / fx
        sy = max(0.0, (y1 - y0)) * z / fy
        sz = min(sx, sy)  # depth extent unknown — approximate with the smaller side

        crop_url: str | None = None
        if out_path is not None and 0 <= rep_frame < len(frames):
            try:
                saved = _save_crop(frames[rep_frame], box_px, out_path, int(object_id))
            except Exception:  # noqa: BLE001 — crops are best-effort
                saved = False
            if saved and request_id is not None:
                crop_url = f"/artifacts/{request_id}/crops/{int(object_id)}.png"

        glb_url = resolve_glb(label) if label else None

        placements.append(
            {
                "object_id": int(object_id),
                "label": str(label),
                "position": [float(pos_x), float(pos_y), float(pos_z)],
                "scale": [float(sx), float(sy), float(sz)],
                "yaw_deg": 0.0,
                "glb_url": glb_url,
                "source": "polypizza" if glb_url else "none",
                "crop_url": crop_url,
            }
        )

    return placements
