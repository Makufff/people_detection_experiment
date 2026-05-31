import argparse

from .constants import DEFAULT_VIDEO_PATH, HERE, resolve_video_path

DESCRIPTION = """People counting under a SAME-UNIFORM crowd — YOLO26 + ByteTrack + re-id.

Pipeline:
   frame -> YOLO26/pose + ByteTrack -> feet anchor
         -> motion gap re-association [+ optional face retrieval]
         -> homography to top-down floor view
         -> count distinct identities dwelling in the entrance floor zone.
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_path", nargs="?",
                    help="input video path; falls back to solution/video/entrance.mov")
    ap.add_argument("--video", default=str(DEFAULT_VIDEO_PATH),
                    help="input video path; same as positional video_path")
    ap.add_argument("--output", default=str(HERE / "result.mp4"))
    ap.add_argument("--model", default="yolo26l.pt",
                    help="YOLO26 person detector. Any ultralytics model works.")
    ap.add_argument("--pose", action="store_true",
                    help="use pose keypoints for the feet anchor and ignore "
                         "upper-body-only/arm detections when triggering the zone")
    ap.add_argument("--pose-model", default="yolo26l-pose.pt",
                    help="[--pose] the pose checkpoint to use")
    ap.add_argument("--tracker", default=str(HERE / "bytetrack_tuned.yaml"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", default=str(HERE / "tracks_cache.csv"),
                    help="per-frame box cache; reused if present, else created")
    ap.add_argument("--retrack", action="store_true",
                    help="ignore an existing cache and re-run the tracker")
    ap.add_argument("--min-in-frames", type=int, default=8,
                    help="dwell frames inside the entrance floor zone before counting")
    ap.add_argument("--no-stitch", action="store_true",
                    help="disable motion gap re-association")
    ap.add_argument("--max-gap", type=int, default=90,
                    help="[stitch] max frame gap to bridge between tracklets")
    ap.add_argument("--max-dist", type=float, default=200.0,
                    help="[stitch] max image-px bridge distance")
    ap.add_argument("--turn-gap", type=int, default=25,
                    help="[stitch] short-gap proximity bridge for turn-arounds")
    ap.add_argument("--face-embeddings", default=None,
                    help="path to face_embeddings.npz to add face retrieval re-id")
    ap.add_argument("--face-sim", type=float, default=0.5,
                    help="[--face-embeddings] cosine threshold")
    ap.add_argument("--direction", action="store_true",
                    help="also report ENTER/EXIT split")
    ap.add_argument("--line-y", type=float, default=0.50,
                    help="[--direction] counting line as a fraction of BEV height")
    ap.add_argument("--min-travel", type=float, default=0.25,
                    help="[--direction] through-motion gate")
    ap.add_argument("--sweep", action="store_true",
                    help="re-count across stitch settings and dwell thresholds")
    ap.add_argument("--log-csv", default=None)
    ap.add_argument("--preview", action="store_true",
                    help="draw the entrance floor zone on a frame, then exit")
    ap.add_argument("--preview-frame", type=int, default=1200)
    ap.add_argument("--no-video", action="store_true",
                    help="count only; skip writing the annotated video")
    ap.add_argument("--crf", type=int, default=28,
                    help="H.264 quality for the re-encoded output")
    ap.add_argument("--crop-dir", default=str(HERE / "counted_crops"),
                    help="folder to save one crop per counted person")
    ap.add_argument("--no-crop", action="store_true",
                    help="do not save per-person crop images")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args(argv)
    if args.video_path:
        args.video = args.video_path
    args.video = str(resolve_video_path(args.video))
    return args
