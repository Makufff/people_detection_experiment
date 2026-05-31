# Solution — People Counting under a Same-Uniform Crowd

Count people who pass through a busy entrance where **everyone wears the same
uniform**. The same pipeline can report either:

| mode | command switch | meaning |
|---|---|---|
| **unique count** | default | merge broken tracklets with re-id, then count each identity once |
| **non-unique / raw tracker count** | `--no-stitch` | skip re-id and count raw ByteTrack track IDs that enter the zone |

Self-contained pipeline:

**YOLO26 (detect) → ByteTrack (track) → optional re-id → floor-zone count**

```bash
# from solution/ (auto-downloads yolo26l.pt, ~53 MB, first run)
uv run python -m pkg.pipeline --device 0 --no-display
#   -> solution/result.mp4        (labelled video, H.264)
#   -> solution/counted_crops/    (one crop image per counted person)
#   -> prints: PEOPLE IN VIDEO (unique) = N
```

For the preset entrypoint with pose enabled and the bundled sample video, run
from the repo root:

```bash
uv run python solution/main.py
```

Or pass a custom video path. Relative names also fall back to
`solution/video/`, so `entrance.mov` resolves to `solution/video/entrance.mov`
when it is not found in the current directory.

```bash
uv run python solution/main.py /path/to/video.mp4
uv run python solution/main.py entrance.mov
```

For full CLI control, run from `solution/`:

```bash
# unique count, default re-id ON
uv run python -m pkg.pipeline --video video/entrance.mov --device 0 --no-display

# non-unique/raw tracker count, re-id OFF
uv run python -m pkg.pipeline --video video/entrance.mov --no-stitch --device 0 --no-display
```

---

## Unique vs non-unique

The counter always uses the same floor-zone rule: a track must have reliable
feet inside the entrance zone for at least `--min-in-frames` frames.

The difference is only the identity source:

| mode | identity source | use when |
|---|---|---|
| **unique** | track IDs after motion re-id stitching, plus optional face re-id | you want approximate distinct people |
| **non-unique / raw** | original ByteTrack IDs before stitching | you want a baseline showing tracker fragmentation |

Turn re-id on/off like this:

```bash
# unique: re-id ON by default
uv run python -m pkg.pipeline --video entrance.mov

# non-unique/raw: re-id OFF
uv run python -m pkg.pipeline --video entrance.mov --no-stitch
```

Important: `--no-stitch` is not a frame-by-frame occupancy count. It still
counts one event per raw tracker ID that dwells in the floor zone. If one real
person is split into three tracker IDs, non-unique mode may count that person
three times.

## Why same-uniform is the hard part — and how we beat it

Identical clothing makes the usual fix fail: **clothing-appearance Re-ID
embeddings collapse** (everyone looks alike), so a person who flickers or is
briefly occluded gets a new track ID (ID churn) and is **counted twice**. The
dark doorway amplifies this (detections blink on/off).

We never use clothing. Re-id is done with **two cues that survive identical
uniforms**:

| Cue | How | Role |
|---|---|---|
| **Motion** (gap re-association) | reconnect a track that ends to one that starts shortly after near its velocity-predicted spot — pure geometry+time | **primary**, ON by default |
| **Motion, turn-around** (`--turn-gap`) | for short gaps also bridge by raw last-position proximity, so a person who **turns around** (walks in facing away, then back out facing the camera) is reconnected instead of counted twice | **primary**, ON by default |
| **Face** (AdaFace retrieval) | embed faces, merge identities whose faces match (face ≠ clothing → not destroyed by the uniform) | **optional** (`--face-embeddings`) |

> **Turn-around (front/back) double-counting:** when someone reverses direction
> their velocity flips, so a plain velocity-prediction bridge overshoots and the
> "facing away" half and the "facing camera" half become two identities. The
> proximity bridge for short gaps fixes this (measured: 39 such split pairs → 33;
> count 77 → 72). Tune with `--turn-gap` (frames; higher catches slower turns,
> too high may merge different people crossing the same spot).

### Better detection: pose-based foot anchor (`--pose`)

YOLO sometimes fires on a *partial* body (an arm, no head/legs). The default
foot anchor — the bbox bottom-centre — is then **wrong** (it is the bottom of an
arm, not the feet), so the floor-zone test misfires. With `--pose` we run a
pose model and read the **ankle keypoints** as the true ground-contact point,
falling back to knee→hip extrapolation, and finally to the bbox bottom only when
no lower body is visible (those detections are flagged **unreliable** and are
ignored when triggering the zone).

Measured on the clip: ankles are visible **96%** of the time (the camera sees
most full bodies), so this is reliable here; **508** upper-body-only detections
were correctly excluded from the zone trigger, and pose tracking fragmented less
(169 vs 266 tracklets). Result: **70** (vs 72 with the bbox anchor) — fewer
spurious zone entries from partial boxes. Slower than plain detection (pose head)
but more accurate; recommended when accuracy matters more than speed.

Counting then happens in a **rectified top-down floor zone**: an identity is
counted once its feet dwell on the doorway floor patch for `--min-in-frames`
frames.
People milling deep in the dark interior have their feet *above* the zone, so
their churn fragments never inflate the count.

```
frame ─► YOLO26 + ByteTrack ─► feet point (bbox bottom-centre or pose ankle)
      ─► optional motion gap re-association  [+ optional face retrieval]
      ─► 4-point homography → top-down floor view
      ─► count identities dwelling in the ENTRANCE FLOOR ZONE
```

---

## Result on the sample clip (`video/entrance.mov`)

| | count | note |
|---|---|---|
| non-unique/raw (`--no-stitch`) | ~97–100 | ID churn over-counts because one person can become multiple track IDs |
| motion re-id, no turn fix | 77 | turn-arounds still double-counted |
| **motion re-id + turn-around (default)** | **72** | YOLO26: 266 fragments → ~140 people → 72 in the entrance zone |
| **+ pose foot anchor (`--pose`)** | **70** | ankle-based feet + drops 508 upper-body-only triggers; tracks cleaner (169 fragments) |
| + face retrieval re-id | (off) | faces too small/turned-away on this wide CCTV — see below |

Honest range with the dedup strength swept: **~60–85** (`--turn-gap` 0→77,
25→72, 45→68). No frame-level ground truth, so we report **72** at the default
settings and a range, rather than a single number on faith.

---

## Options

```bash
--model yolo26l.pt        # detector (yolo26{n,s,m,l,x}.pt; large = best recall)
--pose                    # use pose model: ankle keypoints as the foot anchor,
                          #   and ignore upper-body-only (arm) detections in the zone
--max-gap / --max-dist    # motion re-id strength (↑ merges more fragments)
--turn-gap 25             # turn-around (front/back) re-id window; ↑ catches slower turns
--no-stitch               # disable motion re-id; reports raw/non-unique tracker count
--min-in-frames 8         # dwell frames in the zone before counting
--sweep                   # re-count across re-id strengths instantly (uses cache)
--preview                 # draw the floor zone on one frame and exit
--face-embeddings F.npz   # turn ON face retrieval re-id (see below)
--no-crop / --no-video    # skip crop images / skip the annotated video
--direction               # also report ENTER/EXIT split (not the headline)
```

The first run does one detection pass and caches boxes to
`solution/tracks_cache.csv`; afterwards `--sweep`, unique re-counts, and
`--no-stitch` raw re-counts reuse the same cache. Re-run detection with
`--retrack` after changing detector settings such as `--model`, `--pose`,
`--imgsz`, `--conf`, or `--iou`.

---

## Optional: enable FACE retrieval re-id

Face is the only appearance cue that survives a same-uniform crowd. To use it,
build per-person face embeddings with the bundled `face_embedding/` service
(MTCNN detect → ArcFace align → AdaFace IR-50 embed), then point the counter at
the resulting file:

```bash
cd face_embedding && uv run --with onnxruntime python extract_face_embeddings.py
cd ../solution && uv run python -m pkg.pipeline --face-embeddings ../face_embeddings.npz --face-sim 0.5
```

**On this particular footage it is OFF by default and not recommended**: the
wide CCTV yields faces that are mostly small / dark / turned away, so the
embeddings do not separate people reliably (enabling it over-merges and *under*-
counts). The hook is provided for deployments with clearer, more frontal faces.
See `../FACE_REID.md` for the full measured evidence.

---

## Files
| file | purpose |
|---|---|
| `main.py` | entrypoint; accepts an optional custom video path |
| `pkg/cli.py` | command-line options and defaults |
| `pkg/tracking.py` | YOLO/pose tracking, body-part gate, cache loading |
| `pkg/reid.py` | motion stitch + optional face identity merge |
| `pkg/counting.py` | zone count and optional direction count |
| `pkg/geometry.py` | homography, BEV transform, zone test |
| `pkg/output.py` | preview images, annotated video, counted crops |
| `pkg/pipeline.py` | high-level orchestration of the stages |
| `bytetrack_tuned.yaml` | ByteTrack config tuned to reduce ID churn |
| `result.mp4` | annotated output video |
| `counted_crops/` | one crop per counted person, in count order |
