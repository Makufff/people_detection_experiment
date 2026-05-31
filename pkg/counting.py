import numpy as np

from .constants import BEV_H
from .geometry import in_zone


def smooth(values, k=5):
    if len(values) < 2:
        return values
    kern = np.ones(min(k, len(values))) / min(k, len(values))
    return np.convolve(values, kern, mode="same")


def count_unique(tracks_bev, min_in_frames):
    """Count distinct identities whose reliable feet dwell in the floor zone."""
    count = 0
    log = []
    for tid, seq in tracks_bev.items():
        hits = [p[0] for p in seq if in_zone(p[1], p[2]) and p[3]]
        if len(hits) >= min_in_frames:
            count += 1
            log.append((hits[min_in_frames - 1], tid))
    log.sort()
    return count, log


def count_bev(tracks_bev, line_y_frac, min_travel_frac, min_frames):
    """Count through-traffic split by direction."""
    line_y = line_y_frac * BEV_H
    min_travel = min_travel_frac * BEV_H
    count_in = count_out = 0
    log = []
    for tid, seq in tracks_bev.items():
        seq = [p for p in seq if p[3]]
        if len(seq) < min_frames:
            continue
        ys = smooth(np.array([p[2] for p in seq], dtype=np.float32))
        travel = float(ys.max() - ys.min())
        if travel < min_travel:
            continue
        if not (ys.min() < line_y < ys.max()):
            continue
        net = float(ys[-1] - ys[0])
        direction = "out" if net > 0 else "in"
        rel = ys - line_y
        cross_frame = seq[-1][0]
        for i in range(1, len(rel)):
            if rel[i - 1] * rel[i] < 0:
                cross_frame = seq[i][0]
                break
        if direction == "in":
            count_in += 1
        else:
            count_out += 1
        log.append((cross_frame, tid, direction))
    log.sort()
    return count_in + count_out, count_in, count_out, log

