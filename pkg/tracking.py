from collections import defaultdict

import numpy as np
import supervision as sv
from ultralytics import YOLO

from .constants import (
    KP_ANKLE,
    KP_CONF,
    KP_ELBOW,
    KP_HIP,
    KP_KNEE,
    KP_SHOULDER,
    KP_WRIST,
    PERSON_CLASS_ID,
)
from .geometry import to_bev


def track_to_cache(args, info):
    """Run YOLO+ByteTrack over the clip, cache the full bbox per detection."""
    model = YOLO(args.model)
    n = 0
    with open(args.cache, "w") as f:
        f.write("frame_idx,track_id,x1,y1,x2,y2\n")
        results = model.track(
            source=args.video, tracker=args.tracker, conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, classes=[PERSON_CLASS_ID],
            stream=True, persist=True, verbose=False,
        )
        for frame_idx, result in enumerate(results):
            det = sv.Detections.from_ultralytics(result)
            det = det[det.class_id == PERSON_CLASS_ID]
            if det.tracker_id is None:
                continue
            for i, tid in enumerate(det.tracker_id):
                x1, y1, x2, y2 = (float(v) for v in det.xyxy[i])
                f.write(f"{frame_idx},{int(tid)},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}\n")
                n += 1
    print(f"  tracked {n} detections across {info.total_frames} frames "
          f"-> cache {args.cache}")
    return load_cache(args.cache)


def visible_keypoints(kpt, indices, conf=KP_CONF):
    c = kpt[:, 2]
    return [i for i in indices if c[i] > conf]


def feet_from_keypoints(kpt, x1, y1, x2, y2):
    """Ground-contact point from body parts, instead of the raw bbox bottom."""
    xy = kpt[:, :2]
    shoulder = visible_keypoints(kpt, KP_SHOULDER)
    hip = visible_keypoints(kpt, KP_HIP)
    knee = visible_keypoints(kpt, KP_KNEE)
    ankle = visible_keypoints(kpt, KP_ANKLE)
    arm = visible_keypoints(kpt, KP_ELBOW + KP_WRIST)

    # Arm-only / side-fragment detections help tracking continuity but must not
    # trigger the floor-zone count because there is no trustworthy foot anchor.
    has_body_core = bool(hip or shoulder)
    if ankle and has_body_core:
        return float(xy[ankle, 0].mean()), float(xy[ankle, 1].mean()), 1
    if knee and hip:
        ky, hy = xy[knee, 1].mean(), xy[hip, 1].mean()
        return float(xy[knee, 0].mean()), float(ky + (ky - hy)), 1
    if arm and not has_body_core:
        return (x1 + x2) / 2.0, y2, 0
    return (x1 + x2) / 2.0, y2, 0


def track_to_cache_pose(args, info):
    """Track with a pose model and cache keypoint-derived feet anchors."""
    model = YOLO(args.model)
    n = nun = 0
    with open(args.cache, "w") as f:
        f.write("frame_idx,track_id,x1,y1,x2,y2,feet_x,feet_y,reliable\n")
        results = model.track(
            source=args.video, tracker=args.tracker, conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device, classes=[PERSON_CLASS_ID],
            stream=True, persist=True, verbose=False,
        )
        for frame_idx, result in enumerate(results):
            if result.boxes is None or result.boxes.id is None or result.keypoints is None:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            kpts = result.keypoints.data.cpu().numpy()
            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                fx, fy, rel = feet_from_keypoints(kpts[i], x1, y1, x2, y2)
                f.write(f"{frame_idx},{tid},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},"
                        f"{fx:.1f},{fy:.1f},{rel}\n")
                n += 1
                nun += (rel == 0)
    print(f"  tracked {n} detections (pose); {nun} flagged unreliable feet "
          f"(upper-body only) -> cache {args.cache}")
    return load_cache(args.cache)


def load_cache(path):
    """Return rows as (frame, tid, feet_x, feet_y, box_h, x1, y1, x2, y2, reliable)."""
    rows = []
    with open(path) as f:
        header = next(f).strip().split(",")
        has_pose = "reliable" in header
        has_bbox = "x1" in header
        for ln in f:
            v = ln.strip().split(",")
            fr, tid = int(v[0]), int(v[1])
            if has_pose:
                x1, y1, x2, y2 = (float(t) for t in v[2:6])
                fx, fy = float(v[6]), float(v[7])
                rel = int(v[8])
                rows.append((fr, tid, fx, fy, y2 - y1, x1, y1, x2, y2, rel))
            elif has_bbox:
                x1, y1, x2, y2 = (float(t) for t in v[2:6])
                rows.append((fr, tid, (x1 + x2) / 2.0, y2, y2 - y1, x1, y1, x2, y2, 1))
            else:
                fx, fy, bh = (float(t) for t in v[2:5])
                rows.append((fr, tid, fx, fy, bh, None, None, None, None, 1))
    return rows


def build_tracks_raw(rows):
    """rows -> {tid: [(frame, feet_x, feet_y), ...]} in image px, sorted."""
    by_tid = defaultdict(list)
    for r in rows:
        by_tid[r[1]].append((r[0], r[2], r[3]))
    for tid in by_tid:
        by_tid[tid].sort(key=lambda t: t[0])
    return by_tid


def build_tracks_bev(rows, H):
    """rows -> {tid: [(frame, bev_x, bev_y, reliable), ...]} sorted by frame."""
    by_tid = defaultdict(list)
    feet = np.array([(r[2], r[3]) for r in rows], dtype=np.float32)
    bev = to_bev(feet, H) if len(feet) else np.empty((0, 2))
    for r, (bx, by) in zip(rows, bev):
        rel = r[9] if len(r) > 9 else 1
        by_tid[r[1]].append((r[0], float(bx), float(by), int(rel)))
    for tid in by_tid:
        by_tid[tid].sort(key=lambda t: t[0])
    return by_tid

