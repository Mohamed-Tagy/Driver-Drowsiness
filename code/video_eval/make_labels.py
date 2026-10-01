"""
Build <session>_labels.csv from the independent eye reference and the
scenario windows.

Inputs (in --dir, per session prefix such as P01):
    <s>_reference.csv  per-frame MediaPipe eyelid closure (reference_eyes.py)
    <s>_segments.csv   scenario windows: start_s,end_s,type (written by hand
                       from the session's spoken instructions; types such as
                       baseline, closure_phase, head_tilt_left, head_down,
                       slow_blinks, half_closed, yawn, rub_eyes, sunglasses)
Output:
    <s>_labels.csv     start_s,end_s,type for score_events.py:
        closure        each eyelid closure >= MIN_CUED_S inside closure_phase
        blink          every other closure shorter than MAX_BLINK_S in
                       baseline/closure_phase (spontaneous blinks)
        long_closure   uncued closures >= MAX_BLINK_S in baseline windows
        <scenario>     every scenario window except closure_phase
    <s>_events.png     mid-closure thumbnail of every labelled closure, for
                       visual verification (needs --video)

Closed frame: mean MediaPipe eyeBlink score > THRESH. Runs of closed frames
separated by <= GAP frames are merged; runs shorter than MIN_FRAMES dropped.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

THRESH = 0.5
GAP = 3
MIN_FRAMES = 2
MIN_CUED_S = 0.3
MAX_BLINK_S = 0.5


def closures(ref):
    t = np.array([float(r["t"]) for r in ref])
    score = np.array([(float(r["blink_l"]) + float(r["blink_r"])) / 2
                      if r["face"] == "1" else 0.0 for r in ref])
    closed = score > THRESH
    runs, start = [], None
    for i, c in enumerate(closed):
        if c and start is None:
            start = i
        elif not c and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(closed)])
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= GAP:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    dt = np.median(np.diff(t)) if len(t) > 1 else 1 / 30
    return [(t[a], t[a] + (b - a) * dt) for a, b in merged if b - a >= MIN_FRAMES]


def inside(iv, seg):
    mid = (iv[0] + iv[1]) / 2
    return seg[0] <= mid <= seg[1]


def thumbnails(video, events, out, resize=(1280, 720)):
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tiles = []
    for s, e, kind in events:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((s + e) / 2 * fps))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(fr, resize)
        h, w = fr.shape[:2]
        tile = cv2.resize(fr[int(0.1 * h):int(0.6 * h), int(0.3 * w):int(0.7 * w)], (256, 180))
        cv2.putText(tile, f"{s:.1f}s {e - s:.1f}s {kind[:2]}", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(tile)
    if not tiles:
        return
    cols = 8
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out), grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="video_eval/recordings")
    ap.add_argument("--session", required=True)
    ap.add_argument("--video", help="source video, for verification thumbnails")
    a = ap.parse_args()
    d = Path(a.dir)
    ref = list(csv.DictReader(open(d / f"{a.session}_reference.csv")))
    segs = [(float(r["start_s"]), float(r["end_s"]), r["type"])
            for r in csv.DictReader(open(d / f"{a.session}_segments.csv"))]
    phase = [s for s in segs if s[2] == "closure_phase"]
    normal = [s for s in segs if s[2] in ("baseline", "closure_phase")]
    rows, events = [], []
    for s, e in closures(ref):
        dur = e - s
        if any(inside((s, e), p) for p in phase) and dur >= MIN_CUED_S:
            rows.append((s, e, "closure"))
            events.append((s, e, "closure"))
        elif dur < MAX_BLINK_S and any(inside((s, e), p) for p in normal):
            rows.append((s, e, "blink"))
        elif any(inside((s, e), p) for p in normal):
            # uncued closure of >= MAX_BLINK_S (long blink) during natural gaze
            rows.append((s, e, "long_closure"))
    rows += [s for s in segs if s[2] != "closure_phase"]
    rows.sort()
    with open(d / f"{a.session}_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "type"])
        for s, e, k in rows:
            w.writerow([f"{s:.3f}", f"{e:.3f}", k])
    n_cl = sum(k == "closure" for _, _, k in rows)
    n_bl = sum(k == "blink" for _, _, k in rows)
    print(f"{a.session}: {n_cl} cued closures, {n_bl} spontaneous blinks, "
          f"{len(segs)} scenario windows")
    if a.video:
        thumbnails(a.video, events, d / f"{a.session}_events.png")


if __name__ == "__main__":
    main()
