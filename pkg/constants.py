from pathlib import Path

PERSON_CLASS_ID = 0

HERE = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_PATH = HERE / "video" / "entrance.mov"


def resolve_video_path(video):
    path = Path(video).expanduser()
    if path.exists() or path.is_absolute():
        return path

    fallback = HERE / "video" / path
    if fallback.exists():
        return fallback

    return path

# Four floor points (x, y as ratios of frame w, h) bounding the doorway floor
# patch in the perspective image. Order: far-left, far-right, near-right,
# near-left.
SRC_QUAD = [
    (0.12, 0.42),
    (0.50, 0.42),
    (0.58, 0.97),
    (0.05, 0.97),
]

# BEV canvas size in pixels. The src quad maps onto this whole rectangle.
BEV_W, BEV_H = 600, 800

# COCO keypoint indices used for the ground-contact anchor / body-part gate.
KP_SHOULDER = (5, 6)
KP_ELBOW = (7, 8)
KP_WRIST = (9, 10)
KP_ANKLE = (15, 16)
KP_KNEE = (13, 14)
KP_HIP = (11, 12)
KP_CONF = 0.5

COUNTED_COL = (0, 220, 0)    # green  = this person already counted
PENDING_COL = (0, 220, 220)  # yellow = feet on the zone, dwell not yet satisfied
IDLE_COL = (190, 190, 190)   # grey   = present elsewhere
PARTIAL_COL = (0, 120, 255)  # orange = partial body, anchor not trusted
