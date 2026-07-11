"""SAM 2.1 rigid-object mask extraction.

Frame 0 is auto-segmented with ``SAM2AutomaticMaskGenerator``; the top-K
largest masks are then promoted to object prompts for
``SAM2VideoPredictor`` and propagated across every frame.

When torch or ``sam2`` are not importable — local dev without heavy ML
deps — this module returns empty per-frame mask lists. Downstream stages
tolerate empty segmentation (masks are informational, not load-bearing
for the physics fit).

Docs referenced:
- https://github.com/facebookresearch/sam2
- https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py
- https://github.com/facebookresearch/sam2/blob/main/sam2/build_sam.py
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
_SAM2_CKPT_NAME = "sam2.1_hiera_small.pt"
_SAM2_CKPT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
)

_HANDLE: dict[str, Any] | None = None


def _ensure_checkpoint(weights_dir: str) -> str:
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(weights_dir, _SAM2_CKPT_NAME)
    if not os.path.exists(ckpt_path):
        print(f"[segment] downloading SAM 2.1 checkpoint -> {ckpt_path}")
        urllib.request.urlretrieve(_SAM2_CKPT_URL, ckpt_path)
    return ckpt_path


def _try_load(weights_dir: str = "/weights/sam2") -> dict[str, Any] | None:
    if not _HAS_TORCH:
        return None
    try:
        return load_segmenter(weights_dir)
    except Exception:  # pragma: no cover - defensive against missing sam2
        return None


def load_segmenter(weights_dir: str = "/weights/sam2") -> dict[str, Any]:
    """Build the SAM 2.1 auto mask generator + video predictor."""

    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2, build_sam2_video_predictor

    ckpt = _ensure_checkpoint(weights_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_model = build_sam2(_SAM2_CONFIG, ckpt, device=device)
    mask_gen = SAM2AutomaticMaskGenerator(
        model=image_model,
        points_per_side=16,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=400,
    )
    video_predictor = build_sam2_video_predictor(_SAM2_CONFIG, ckpt, device=device)
    return {
        "mask_gen": mask_gen,
        "video_predictor": video_predictor,
        "device": device,
    }


def prefetch(weights_dir: str = "/weights/sam2") -> None:
    global _HANDLE
    if _HANDLE is None:
        _HANDLE = _try_load(weights_dir)


def segment(
    frames: list[np.ndarray],
    handle: dict[str, Any] | None = None,
    max_objects: int = 4,
) -> list[list[dict]]:
    """Return per-frame lists of ``{"mask", "label", "score"}`` dicts.

    When SAM 2 is unavailable this returns ``[[] for _ in frames]`` — an
    empty-but-shape-correct result the rest of the pipeline tolerates.
    """

    if not frames:
        return []

    if handle is None:
        handle = _HANDLE
    if handle is None:
        handle = _try_load()
    if handle is None:
        return [[] for _ in frames]

    import cv2
    import torch

    device = handle["device"]

    frame0 = frames[0]
    with torch.inference_mode():
        raw_masks = handle["mask_gen"].generate(frame0)

    raw_masks.sort(key=lambda m: -m["area"])
    kept = raw_masks[:max_objects]
    per_frame_masks: list[list[dict]] = [[] for _ in frames]
    if not kept:
        return per_frame_masks

    scratch = tempfile.mkdtemp(prefix="kepler-sam2-")
    try:
        for i, f in enumerate(frames):
            p = os.path.join(scratch, f"{i:05d}.jpg")
            cv2.imwrite(p, cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

        predictor = handle["video_predictor"]
        autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        with torch.inference_mode(), torch.autocast(
            device_type=device, dtype=autocast_dtype
        ):
            state = predictor.init_state(video_path=scratch)

            for obj_id, m in enumerate(kept):
                x, y, w, h = m["bbox"]
                box = np.array([x, y, x + w, y + h], dtype=np.float32)
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    box=box,
                )

            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                for i, oid in enumerate(obj_ids):
                    mask = (
                        (mask_logits[i] > 0.0)
                        .squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.bool_)
                    )
                    per_frame_masks[int(frame_idx)].append(
                        {
                            "mask": mask,
                            "label": f"obj_{int(oid)}",
                            "score": 1.0,
                        }
                    )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    return per_frame_masks
