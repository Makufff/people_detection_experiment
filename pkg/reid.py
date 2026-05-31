import numpy as np

from collections import defaultdict


def stitch_roots(tracks_xy, max_gap, max_dist, turn_gap=25):
    """Motion-only gap re-association for same-uniform ID churn.

    For short gaps we also accept raw last-position proximity. That handles a
    person who turns around: the velocity prediction points the wrong way, but
    the new track starts near the last known position.
    """
    parent = {tid: tid for tid in tracks_xy}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    def end_velocity(seq):
        if len(seq) < 2:
            return 0.0, 0.0
        n = min(5, len(seq))
        f0, x0, y0 = seq[-n]
        f1, x1, y1 = seq[-1]
        df = max(1, f1 - f0)
        return (x1 - x0) / df, (y1 - y0) / df

    by_start = sorted(tracks_xy.items(), key=lambda kv: kv[1][0][0])
    consumed_ends = set()
    for tidB, seqB in by_start:
        fB, bx, by = seqB[0]
        best, best_d = None, max_dist
        for tidA, seqA in tracks_xy.items():
            if tidA == tidB or tidA in consumed_ends:
                continue
            fA, ax, ay = seqA[-1]
            gap = fB - fA
            if gap <= 0 or gap > max_gap:
                continue
            vx, vy = end_velocity(seqA)
            px, py = ax + vx * gap, ay + vy * gap
            d_pred = float(np.hypot(px - bx, py - by))
            d_pos = float(np.hypot(ax - bx, ay - by))
            d = min(d_pred, d_pos) if gap <= turn_gap else d_pred
            if d < best_d and find(tidA) != find(tidB):
                best_d, best = d, tidA
        if best is not None:
            union(best, tidB)
            consumed_ends.add(best)

    return {tid: find(tid) for tid in tracks_xy}


def face_compose(roots, npz_path, sim):
    """Compose optional face retrieval re-id on top of motion stitch roots."""
    d = np.load(npz_path)
    emb_of = {int(t): e for t, e in zip(d["tids"], d["embs"])}
    groups = defaultdict(list)
    for raw_tid, mroot in roots.items():
        if raw_tid in emb_of:
            groups[mroot].append(emb_of[raw_tid])
    desc = {}
    for mroot, vs in groups.items():
        m = np.mean(np.stack(vs), axis=0)
        desc[mroot] = m / (np.linalg.norm(m) + 1e-8)

    parent = {m: m for m in set(roots.values())}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mids = list(desc)
    merges = 0
    for i, a in enumerate(mids):
        for b in mids[i + 1:]:
            if find(a) != find(b) and float(desc[a] @ desc[b]) >= sim:
                parent[find(b)] = find(a)
                merges += 1
    print(f"  face re-id: {len(desc)} identities had a descriptor, "
          f"{merges} extra merge(s) at cos>={sim}")
    return {raw_tid: find(mroot) for raw_tid, mroot in roots.items()}

