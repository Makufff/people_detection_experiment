"""Entrypoint for running the solution on one video.

Run with the bundled sample video:

    uv run python solution/main.py

Or pass a custom input path:

    uv run python solution/main.py /path/to/video.mp4
"""

from pathlib import Path

from pkg.cli import parse_args
from pkg.pipeline import run


HERE = Path(__file__).resolve().parent

# Output/cache locations. Keeping them under solution/ avoids mixing with the
# older root-level scripts.
OUTPUT_PATH = HERE / "result.mp4"
CACHE_PATH = HERE / "tracks_cache.csv"

# Recommended for this footage: pose gives a better foot anchor and ignores
# arm-only / upper-body-only detections when triggering the floor zone.
USE_POSE = True
WRITE_VIDEO = True
SAVE_CROPS = True
SHOW_WINDOW = False


def main():
    args = parse_args()
    args.output = str(OUTPUT_PATH)
    args.cache = str(CACHE_PATH)
    args.pose = USE_POSE
    args.no_video = not WRITE_VIDEO
    args.no_crop = not SAVE_CROPS
    args.no_display = not SHOW_WINDOW

    if USE_POSE and CACHE_PATH.name == "tracks_cache.csv":
        args.cache = str(CACHE_PATH.with_name("tracks_cache_pose.csv"))

    run(args)


if __name__ == "__main__":
    main()
