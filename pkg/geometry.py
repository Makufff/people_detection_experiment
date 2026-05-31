import cv2
import numpy as np

from .constants import BEV_H, BEV_W, SRC_QUAD


def src_quad_px(w, h):
    return np.array([(x * w, y * h) for x, y in SRC_QUAD], dtype=np.float32)


def dst_quad_px():
    # far edge -> BEV top, near edge -> BEV bottom
    return np.array([(0, 0), (BEV_W, 0), (BEV_W, BEV_H), (0, BEV_H)], dtype=np.float32)


def homography(w, h):
    return cv2.getPerspectiveTransform(src_quad_px(w, h), dst_quad_px())


def to_bev(points_xy, H):
    """points_xy: (N,2) image coords -> (N,2) BEV coords."""
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def in_zone(bx, by):
    """A feet point is on the entrance floor iff it lands inside the BEV canvas."""
    return 0.0 <= bx < BEV_W and 0.0 <= by < BEV_H


def line_in_image(H, line_y_frac):
    """Back-project the horizontal BEV counting line into image coordinates."""
    ly = line_y_frac * BEV_H
    ends = np.array([[0.0, ly], [float(BEV_W), ly]], dtype=np.float32)
    img = to_bev(ends, np.linalg.inv(H))
    return img[0].astype(int), img[1].astype(int)

