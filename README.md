# Solution — People Counting under a Same-Uniform Crowd

Count the distinct people who pass through a busy entrance where **everyone
wears the same uniform**. Self-contained pipeline:

**YOLO26 (detect) → ByteTrack (track) → re-id (motion + optional face) → floor-zone unique count**

```bash
# from the repo root (auto-downloads yolo26l.pt, ~53 MB, first run)
uv run python solution/count_people.py --device 0 --no-display
#   -> solution/result.mp4        (labelled video, H.264)
#   -> solution/counted_crops/    (one crop image per counted person)
#   -> prints: PEOPLE IN VIDEO (unique) = N
```

For the entrypoint with the bundled sample video, run:

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

---

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

Counting then happens in a **rectified top-down floor zone**: a person is counted
once their feet dwell on the doorway floor patch for `--min-in-frames` frames.
People milling deep in the dark interior have their feet *above* the zone, so
their churn fragments never inflate the count.

```
frame ─► YOLO26 + ByteTrack ─► feet point (bbox bottom-centre)
      ─► motion gap re-association  [+ optional face retrieval]   ← solves same-uniform
      ─► 4-point homography → top-down floor view
      ─► count distinct identities dwelling in the ENTRANCE FLOOR ZONE
```

---

## Result on the sample clip (`video/entrance.mov`)

| | count | note |
|---|---|---|
| no re-id (raw zone presence) | ~97–100 | ID churn over-counts |
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
--min-in-frames 8         # dwell frames in the zone before counting
--sweep                   # re-count across re-id strengths instantly (uses cache)
--preview                 # draw the floor zone on one frame and exit
--face-embeddings F.npz   # turn ON face retrieval re-id (see below)
--no-crop / --no-video    # skip crop images / skip the annotated video
--direction               # also report ENTER/EXIT split (not the headline)
```

The first run does one detection pass and caches boxes to
`solution/tracks_cache.csv`; afterwards `--sweep` and re-counts are instant.
Re-run detection with `--retrack` (e.g. after changing `--model`).

---

## Optional: enable FACE retrieval re-id

Face is the only appearance cue that survives a same-uniform crowd. To use it,
build per-person face embeddings with the bundled `face_embedding/` service
(MTCNN detect → ArcFace align → AdaFace IR-50 embed), then point the counter at
the resulting file:

```bash
cd face_embedding && uv run --with onnxruntime python extract_face_embeddings.py
cd .. && uv run python solution/count_people.py --face-embeddings face_embeddings.npz --face-sim 0.5
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
| `count_people.py` | CLI entrypoint; keeps the old command stable |
| `people_counter/cli.py` | command-line options and defaults |
| `people_counter/tracking.py` | YOLO/pose tracking, body-part gate, cache loading |
| `people_counter/reid.py` | motion stitch + optional face identity merge |
| `people_counter/counting.py` | unique-zone count and optional direction count |
| `people_counter/geometry.py` | homography, BEV transform, zone test |
| `people_counter/output.py` | preview images, annotated video, counted crops |
| `people_counter/pipeline.py` | high-level orchestration of the stages |
| `bytetrack_tuned.yaml` | ByteTrack config tuned to reduce ID churn |
| `result.mp4` | annotated output video |
| `counted_crops/` | one crop per counted person, in count order |
