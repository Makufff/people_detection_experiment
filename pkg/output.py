import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from .constants import (
    BEV_H,
    BEV_W,
    COUNTED_COL,
    IDLE_COL,
    PARTIAL_COL,
    PENDING_COL,
)
from .geometry import homography, in_zone, src_quad_px, to_bev


def preview(args, info):
    w, h = info.resolution_wh
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.preview_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read preview frame")
    H = homography(w, h)
    quad = src_quad_px(w, h).astype(np.int32)
    ov = frame.copy()
    cv2.fillPoly(ov, [quad], (0, 200, 0))
    frame = cv2.addWeighted(ov, 0.3, frame, 0.7, 0)
    cv2.polylines(frame, [quad], True, (0, 255, 0), 3)
    cv2.imwrite("preview_bev_src.jpg", frame)

    bev = cv2.warpPerspective(_read_frame(args), H, (BEV_W, BEV_H))
    ly = int(args.line_y * BEV_H)
    cv2.line(bev, (0, ly), (BEV_W, ly), (0, 255, 255), 3)
    cv2.imwrite("preview_bev_top.jpg", bev)
    print("wrote preview_bev_src.jpg (quad on frame) and preview_bev_top.jpg (top-down + line)")


def _read_frame(args):
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.preview_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read preview frame")
    return frame


def render_video(args, info, rows, H, log, total):
    """Annotated perspective video for the unique people count."""
    w, h = info.resolution_wh
    quad = src_quad_px(w, h).astype(np.int32)

    counted_frame = {tid: fr for fr, tid in log}
    feet = np.array([(r[2], r[3]) for r in rows], dtype=np.float32)
    bev = to_bev(feet, H) if len(feet) else np.empty((0, 2))
    by_frame = defaultdict(list)
    for r, (bx, by) in zip(rows, bev):
        reliable = r[9] if len(r) > 9 else 1
        by_frame[r[0]].append((r[1], r[2], r[3], in_zone(bx, by), reliable))

    counted_ids, running = set(), 0
    frame_gen = sv.get_video_frames_generator(args.video)
    with sv.VideoSink(args.output, video_info=info) as sink:
        for fr, frame in enumerate(frame_gen):
            for tid, cf in counted_frame.items():
                if cf == fr and tid not in counted_ids:
                    counted_ids.add(tid)
                    running += 1

            cv2.polylines(frame, [quad], True, (0, 200, 0), 2)
            cv2.putText(frame, "ENTRANCE FLOOR ZONE", (quad[0][0], quad[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)

            for tid, fx, fy, inz, reliable in by_frame.get(fr, []):
                if not reliable:
                    col = PARTIAL_COL
                elif tid in counted_ids:
                    col = COUNTED_COL
                elif inz:
                    col = PENDING_COL
                else:
                    col = IDLE_COL
                cv2.circle(frame, (int(fx), int(fy)), 5, col, -1)
                cv2.putText(frame, f"#{tid}", (int(fx) + 6, int(fy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            cv2.rectangle(frame, (10, h - 70), (470, h - 16), (0, 0, 0), -1)
            cv2.putText(frame, f"PEOPLE COUNTED: {running} / {total}", (20, h - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, COUNTED_COL, 2)
            sink.write_frame(frame)
            if not args.no_display:
                cv2.imshow("bev-count", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    cv2.destroyAllWindows()
    reencode_h264(args.output, args.crf)
    print(f"  annotated video -> {Path(args.output).resolve()}")


def reencode_h264(path, crf):
    """Re-encode the sv mp4v output to H.264 in place."""
    if shutil.which("ffmpeg") is None:
        print("  (ffmpeg not found — keeping mp4v video; install ffmpeg for smaller files)")
        return
    src, tmp = Path(path), Path(path).with_suffix(".h264.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    try:
        subprocess.run(cmd, check=True)
        tmp.replace(src)
    except (subprocess.CalledProcessError, OSError) as e:
        tmp.unlink(missing_ok=True)
        print(f"  (H.264 re-encode skipped: {e}; keeping mp4v)")


def _write_crop(frame, bbox, out_path, pad=0.12):
    """Crop a padded person box from the frame and save it."""
    if any(v is None for v in bbox):
        return False
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    X2, Y2 = min(w, int(x2 + pad * bw)), min(h, int(y2 + pad * bh))
    crop = frame[Y1:Y2, X1:X2]
    if crop.size == 0:
        return False
    cv2.imwrite(str(out_path), crop)
    return True


def save_crops(args, rows, log, crop_dir):
    """Save one cropped image per counted person at the counted frame."""
    bbox_at = {(r[0], r[1]): (r[5], r[6], r[7], r[8]) for r in rows}
    if all(v[0] is None for v in bbox_at.values()):
        print("  (no bbox in cache — run with --retrack to enable crops)")
        return
    need = defaultdict(list)
    for order, (fr, tid) in enumerate(log, 1):
        need[fr].append((order, tid, bbox_at.get((fr, tid), (None,) * 4)))

    out = Path(crop_dir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()

    saved = 0
    for fr, frame in enumerate(sv.get_video_frames_generator(args.video)):
        if fr not in need:
            continue
        for order, tid, bbox in need[fr]:
            if _write_crop(frame, bbox, out / f"{order:03d}_id{tid}_f{fr}.jpg"):
                saved += 1
    print(f"  saved {saved} person crops -> {out.resolve()}/")

