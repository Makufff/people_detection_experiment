from pathlib import Path

import supervision as sv

from .cli import parse_args
from .constants import HERE
from .counting import count_bev, count_unique
from .geometry import homography
from .output import preview, render_video, save_crops
from .reid import face_compose, stitch_roots
from .tracking import (
    build_tracks_bev,
    build_tracks_raw,
    load_cache,
    track_to_cache,
    track_to_cache_pose,
)


def select_tracker(args):
    if not args.pose:
        return track_to_cache
    if args.model == "yolo26l.pt":
        args.model = args.pose_model
    if args.cache == str(HERE / "tracks_cache.csv"):
        args.cache = str(HERE / "tracks_cache_pose.csv")
    return track_to_cache_pose


def load_or_track(args, info):
    tracker_fn = select_tracker(args)
    if args.retrack or not Path(args.cache).exists():
        return tracker_fn(args, info)
    rows = load_cache(args.cache)
    print(f"  loaded {len(rows)} detections from cache {args.cache}")
    return rows


def apply_reid(args, rows):
    stitch_on = not args.no_stitch
    if not stitch_on:
        return rows, False

    raw = build_tracks_raw(rows)
    roots = stitch_roots(raw, args.max_gap, args.max_dist, args.turn_gap)
    n_motion = len(set(roots.values()))
    if args.face_embeddings:
        roots = face_compose(roots, args.face_embeddings, args.face_sim)

    stitched_rows = [(fr, roots[tid], *rest) for (fr, tid, *rest) in rows]
    msg = f"  stitch (re-id dedup): {len(raw)} tracklets -> {n_motion} identities"
    if args.face_embeddings:
        msg += f" -> {len(set(roots.values()))} after face re-id"
    print(msg + f" (max_gap={args.max_gap}, max_dist={args.max_dist}px)")
    return stitched_rows, True


def run_sweep(args, H):
    cache_rows = load_cache(args.cache)
    raw_all = build_tracks_raw(cache_rows)
    print("\nUNIQUE-PEOPLE SWEEP — robustness to re-id (stitch) & dwell:")
    print(f"  {'max_gap':>7} {'max_dist':>8} | dwell>=5  >=8  >=15")
    for mg in (45, 90, 150):
        for md in (180, 250, 350):
            rt = stitch_roots(raw_all, mg, md, args.turn_gap)
            rr = [(fr, rt[t], *rest) for (fr, t, *rest) in cache_rows]
            tb = build_tracks_bev(rr, H)
            print(f"  {mg:>7} {md:>8} |   {count_unique(tb, 5)[0]:>4}  "
                  f"{count_unique(tb, 8)[0]:>4}  {count_unique(tb, 15)[0]:>4}")
    print("  (no-stitch baseline ~ count_zone = 97; lower = more dedup)")


def write_log_csv(path, log):
    with open(path, "w") as f:
        f.write("frame_counted,identity_id\n")
        for fr, tid in log:
            f.write(f"{fr},{tid}\n")
    print(f"  count log -> {path}")


def run(args):
    if not Path(args.video).exists():
        raise SystemExit(f"video not found: {args.video}")

    info = sv.VideoInfo.from_video_path(args.video)
    w, h = info.resolution_wh
    H = homography(w, h)

    if args.preview:
        preview(args, info)
        return

    rows = load_or_track(args, info)
    rows, stitch_on = apply_reid(args, rows)
    tracks_bev = build_tracks_bev(rows, H)

    if args.sweep:
        run_sweep(args, H)
        return

    count, log = count_unique(tracks_bev, args.min_in_frames)

    if args.log_csv:
        write_log_csv(args.log_csv, log)

    if not args.no_crop:
        save_crops(args, rows, log, args.crop_dir)

    if not args.no_video:
        render_video(args, info, rows, H, log, count)

    print("=" * 56)
    print(f"  PEOPLE IN VIDEO (unique, stepped on entrance zone): {count}")
    print(f"  stitch={'on' if stitch_on else 'OFF'}  min_in_frames={args.min_in_frames}")
    print("=" * 56)

    if args.direction:
        tot, ci, co, _ = count_bev(
            tracks_bev, args.line_y, args.min_travel, args.min_in_frames)
        print(f"  [--direction] of the through-traffic: ENTER {ci} / EXIT {co}")


def main(argv=None):
    run(parse_args(argv))
