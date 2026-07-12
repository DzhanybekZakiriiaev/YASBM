"""Object-centric analysis: detect, associate, audit.

This stage turns the blind CoTracker grid into *named objects* with
per-object verdicts:

1. **Detect** — YOLOv8n per frame (COCO classes: person, sports ball,
   cup, cell phone, ...).
2. **Associate** — greedy IoU matching links detections across frames
   into persistent objects.
3. **Membership** — each CoTracker track is assigned to the object whose
   boxes contain it most often.
4. **Ballistic audit** — the object's member-track 3D centroid trajectory
   is fitted with ``kepler_physics.fit``. Self-propelled classes (person,
   car, dog, ...) are *exempt*: muscles and engines legitimately violate
   projectile physics, so flagging them would be a false positive.
5. **Morph audit** — three signals catch generative "morphing":
   - *class flicker*: the detector keeps changing its mind about what
     the object is (phone → wallet → phone);
   - *rigidity violation*: pairwise 3D distances between member tracks
     should stay ~constant on a rigid body;
   - *box jerk*: bounding-box area second-difference — shape pulsing
     that motion can't explain.

Falls back to an empty object list when ultralytics isn't importable
(local dev without ML deps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from ultralytics import YOLO  # noqa: F401

    _HAS_YOLO = True
except ImportError:  # pragma: no cover
    _HAS_YOLO = False

_MODEL: Any = None
_WEIGHTS = "yolov8n.pt"

# COCO classes whose motion is powered — ballistic physics does not apply.
SELF_PROPELLED = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

# Articulated classes get a looser rigidity threshold (limbs move).
ARTICULATED = {"person", "cat", "dog", "horse", "bird", "bear"}

_MIN_DET_CONF = 0.35
_IOU_MATCH_THRESHOLD = 0.30
_MIN_FRAMES_PRESENT = 4  # objects seen fewer frames than this are noise
_MOVING_DISPLACEMENT_M = 0.25


@dataclass
class TrackedObject:
    object_id: int
    label: str  # modal class name
    class_votes: dict = field(default_factory=dict)
    # frame_idx -> (x0, y0, x1, y1) pixel box
    boxes: dict = field(default_factory=dict)
    confidences: dict = field(default_factory=dict)
    member_track_ids: list = field(default_factory=list)


def _get_model() -> Any:
    global _MODEL
    if not _HAS_YOLO:
        return None
    if _MODEL is None:
        _MODEL = YOLO(_WEIGHTS)
    return _MODEL


def prefetch() -> None:
    """Warm-load YOLO weights; call from ``@modal.enter()``."""

    _get_model()


def _iou(a: tuple, b: tuple) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / max(area_a + area_b - inter, 1e-9)


def _detect_all(frames: list[np.ndarray]) -> list[list[dict]]:
    """Per-frame detections: [{box, cls_name, conf}, ...]."""

    model = _get_model()
    if model is None:
        return [[] for _ in frames]

    per_frame: list[list[dict]] = []
    for frame in frames:
        # Ultralytics assumes BGR for raw ndarrays; flip from our RGB.
        results = model.predict(source=frame[..., ::-1], verbose=False, conf=_MIN_DET_CONF)
        dets: list[dict] = []
        if results:
            r = results[0]
            names = r.names
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                dets.append(
                    {
                        "box": tuple(float(v) for v in xyxy),
                        "cls_name": names[int(b.cls[0].item())],
                        "conf": float(b.conf[0].item()),
                    }
                )
        per_frame.append(dets)
    return per_frame


def _associate(per_frame: list[list[dict]]) -> list[TrackedObject]:
    """Greedy IoU matching, frame to frame, into persistent objects."""

    objects: list[TrackedObject] = []
    active: list[TrackedObject] = []
    next_id = 0

    for f_idx, dets in enumerate(per_frame):
        unmatched = list(range(len(dets)))
        # Match existing active objects to this frame's detections.
        for obj in active:
            last_box = obj.boxes.get(f_idx - 1)
            if last_box is None:
                continue
            best_j, best_iou = -1, _IOU_MATCH_THRESHOLD
            for j in unmatched:
                iou = _iou(last_box, dets[j]["box"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                d = dets[best_j]
                obj.boxes[f_idx] = d["box"]
                obj.confidences[f_idx] = d["conf"]
                obj.class_votes[d["cls_name"]] = (
                    obj.class_votes.get(d["cls_name"], 0.0) + d["conf"]
                )
                unmatched.remove(best_j)
        # Unmatched detections spawn new objects.
        for j in unmatched:
            d = dets[j]
            obj = TrackedObject(object_id=next_id, label=d["cls_name"])
            obj.boxes[f_idx] = d["box"]
            obj.confidences[f_idx] = d["conf"]
            obj.class_votes[d["cls_name"]] = d["conf"]
            next_id += 1
            active.append(obj)
            objects.append(obj)

    # Finalise labels by weighted vote; drop blips.
    kept = []
    for obj in objects:
        if len(obj.boxes) < _MIN_FRAMES_PRESENT:
            continue
        obj.label = max(obj.class_votes.items(), key=lambda kv: kv[1])[0]
        kept.append(obj)
    return kept


def _assign_tracks(
    objects: list[TrackedObject],
    tracks_2d: np.ndarray,  # (T, N, 2) pixel coords
) -> None:
    """Assign each track to the object whose boxes contain it most often."""

    t_frames, n_tracks, _ = tracks_2d.shape
    for track_id in range(n_tracks):
        best_obj, best_hits = None, 0
        for obj in objects:
            hits = 0
            for f_idx, box in obj.boxes.items():
                if f_idx >= t_frames:
                    continue
                u, v = tracks_2d[f_idx, track_id]
                x0, y0, x1, y1 = box
                if x0 <= u <= x1 and y0 <= v <= y1:
                    hits += 1
            if hits > best_hits:
                best_hits, best_obj = hits, obj
        # Require presence in at least 30% of the object's observed frames.
        if best_obj is not None and best_hits >= max(2, int(0.3 * len(best_obj.boxes))):
            best_obj.member_track_ids.append(track_id)


def _class_flicker(obj: TrackedObject) -> float:
    """Fraction of confidence-weight cast on classes other than the winner."""

    total = sum(obj.class_votes.values())
    if total <= 0:
        return 0.0
    return 1.0 - obj.class_votes.get(obj.label, 0.0) / total


def _rigidity_violation(
    obj: TrackedObject, tracks_3d: np.ndarray
) -> float:
    """Median coefficient of variation of pairwise 3D distances.

    Rigid bodies keep pairwise distances constant (~0.02 from tracking
    noise). Morphing objects don't. Articulated classes run higher
    legitimately — the caller compensates via thresholds.
    """

    ids = obj.member_track_ids
    if len(ids) < 2:
        return 0.0
    pts = tracks_3d[:, ids, :]  # (T, K, 3)
    cvs: list[float] = []
    k = len(ids)
    for i in range(k):
        for j in range(i + 1, k):
            d = np.linalg.norm(pts[:, i, :] - pts[:, j, :], axis=1)  # (T,)
            mean = float(np.mean(d))
            if mean < 1e-6:
                continue
            cvs.append(float(np.std(d)) / mean)
    return float(np.median(cvs)) if cvs else 0.0


def _box_jerk(obj: TrackedObject) -> float:
    """Normalised second-difference of box area over the frames present."""

    frames = sorted(obj.boxes.keys())
    if len(frames) < 3:
        return 0.0
    areas = np.array(
        [
            (obj.boxes[f][2] - obj.boxes[f][0]) * (obj.boxes[f][3] - obj.boxes[f][1])
            for f in frames
        ],
        dtype=np.float64,
    )
    mean_area = float(np.mean(areas))
    if mean_area < 1e-6:
        return 0.0
    second = np.abs(areas[2:] - 2 * areas[1:-1] + areas[:-2])
    return float(np.median(second)) / mean_area


def _morph_score(obj: TrackedObject, tracks_3d: np.ndarray) -> dict:
    flicker = _class_flicker(obj)
    rigidity = _rigidity_violation(obj, tracks_3d)
    jerk = _box_jerk(obj)

    # Articulated bodies flex legitimately — discount their rigidity signal.
    rigidity_budget = 0.20 if obj.label in ARTICULATED else 0.05
    rigidity_excess = max(0.0, rigidity - rigidity_budget) / max(rigidity_budget, 1e-6)

    # 0..1-ish combined score. Flicker is the strongest generative tell.
    score = min(1.0, 0.5 * flicker / 0.3 + 0.35 * min(rigidity_excess, 2.0) / 2.0 + 0.15 * min(jerk / 0.15, 1.0))
    return {
        "morph_score": float(score),
        "class_flicker": float(flicker),
        "rigidity_cv": float(rigidity),
        "box_jerk": float(jerk),
    }


def _ballistic_audit(
    obj: TrackedObject,
    tracks_3d: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """Fit the ballistic model to the object's member-centroid trajectory."""

    from .physics import fit_trajectory  # reuse the sys.path shim there

    ids = obj.member_track_ids
    if not ids:
        return {"eligible": False, "reason": "no member tracks", "sigma": 0.0}

    centroid = tracks_3d[:, ids, :].mean(axis=1)  # (T, 3)

    # Moving? Path displacement gate — static objects have nothing to audit.
    displacement = float(np.linalg.norm(centroid[-1] - centroid[0]))
    max_dev = float(np.max(np.linalg.norm(centroid - centroid[0], axis=1)))
    if max(displacement, max_dev) < _MOVING_DISPLACEMENT_M:
        return {"eligible": False, "reason": "static", "sigma": 0.0}

    if obj.label in SELF_PROPELLED:
        return {
            "eligible": False,
            "reason": "self-propelled",
            "sigma": 0.0,
        }

    try:
        result = fit_trajectory(
            centroid.astype(np.float64), timestamps.astype(np.float64)
        )
        return {"eligible": True, "reason": "free-motion rigid object", "sigma": float(result.peak_sigma)}
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {"eligible": False, "reason": f"fit failed: {exc}", "sigma": 0.0}


def objects(
    frames: list[np.ndarray],
    tracks_2d: np.ndarray,
    tracks_3d: np.ndarray,
    timestamps: np.ndarray,
) -> list[dict]:
    """Run the full object-centric audit. Returns JSON-ready dicts."""

    if not frames or not _HAS_YOLO:
        return []

    per_frame = _detect_all(frames)
    objs = _associate(per_frame)
    _assign_tracks(objs, tracks_2d)

    height, width = frames[0].shape[:2]
    reports: list[dict] = []
    for obj in objs:
        morph = _morph_score(obj, tracks_3d)
        ballistic = _ballistic_audit(obj, tracks_3d, timestamps)

        sigma = ballistic["sigma"]
        if ballistic["eligible"]:
            if sigma < 3:
                verdict = "consistent"
            elif sigma < 10:
                verdict = "borderline"
            else:
                verdict = "implausible"
        elif ballistic["reason"] == "self-propelled":
            verdict = "agent"
        else:
            verdict = "static"
        if morph["morph_score"] > 0.6:
            verdict = "morphing"

        # Normalized per-frame boxes for the browser overlay.
        boxes_norm = {
            str(f): [
                b[0] / width,
                b[1] / height,
                b[2] / width,
                b[3] / height,
            ]
            for f, b in obj.boxes.items()
        }

        reports.append(
            {
                "object_id": obj.object_id,
                "label": obj.label,
                "frames_present": len(obj.boxes),
                "member_track_ids": obj.member_track_ids,
                "boxes_norm": boxes_norm,
                "ballistic": ballistic,
                **morph,
                "verdict": verdict,
            }
        )

    return reports
