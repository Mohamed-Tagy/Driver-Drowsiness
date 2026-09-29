"""
Record a scripted webcam session for system-level evaluation.

The participant follows on-screen cues ("CLOSE EYES", "OPEN EYES",
"TURN HEAD LEFT", ...). The script saves:
    <out>/<session>.mp4            the raw video (no model runs here)
    <out>/<session>_frames.csv     frame index and capture timestamp
    <out>/<session>_cues.csv       every cue interval, used as draft labels

Cue intervals are only *instructed* behavior. Check and correct them
against the video before scoring (see PROTOCOL.md), because people react
to cues with a delay and blink spontaneously.

Usage:
    python video_eval/record_session.py --participant P01 --condition normal
    python video_eval/record_session.py --participant P01 --condition glasses --camera 1
"""

import argparse
import csv
import random
import time
from pathlib import Path

import cv2

# (cue text, event type written to the cues file, duration in seconds)
BASELINE = [("LOOK AT THE ROAD - blink normally", "baseline", 60.0)]
CLOSURE_SECONDS = [0.5, 1.0, 2.0, 3.0, 5.0]
REPEATS = 3
HEAD_POSES = [("TURN HEAD LEFT (~30 deg)", "head_left", 5.0),
              ("TURN HEAD RIGHT (~30 deg)", "head_right", 5.0),
              ("LOOK DOWN (~20 deg)", "head_down", 5.0),
              ("LOOK UP (~20 deg)", "head_up", 5.0)]


def build_script(seed):
    rng = random.Random(seed)
    closures = [s for s in CLOSURE_SECONDS for _ in range(REPEATS)]
    rng.shuffle(closures)
    script = list(BASELINE)
    for s in closures:
        script.append(("OPEN EYES - look ahead", "open", rng.uniform(6.0, 10.0)))
        script.append((f"CLOSE EYES ({s:g} s)", f"closure_{s:g}s", s))
    script.append(("OPEN EYES - look ahead", "open", 8.0))
    for cue in HEAD_POSES:
        script.append(cue)
        script.append(("OPEN EYES - look ahead", "open", 5.0))
    script.append(("LOOK AT THE ROAD - blink normally", "baseline", 60.0))
    return script


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--participant", required=True)
    p.add_argument("--condition", default="normal",
                   help="e.g. normal, glasses, dim_light")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--out", default="video_eval/recordings")
    p.add_argument("--seed", type=int, default=0, help="Order of the closure cues")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = f"{args.participant}_{args.condition}_{time.strftime('%Y%m%d_%H%M%S')}"
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Cannot read from camera {args.camera}")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(out / f"{session}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    script = build_script(args.seed)
    total = sum(d for _, _, d in script)
    print(f"Session {session}: {len(script)} cues, about {total / 60:.1f} min. "
          "Press Q to abort.")
    frames_f = open(out / f"{session}_frames.csv", "w", newline="")
    cues_f = open(out / f"{session}_cues.csv", "w", newline="")
    fw, cw = csv.writer(frames_f), csv.writer(cues_f)
    fw.writerow(["frame", "t"])
    cw.writerow(["start_s", "end_s", "type", "cue"])

    t0 = time.perf_counter()
    frame_idx, aborted = 0, False
    try:
        for text, kind, dur in script:
            start = time.perf_counter() - t0
            while (now := time.perf_counter() - t0) < start + dur:
                ok, frame = cap.read()
                if not ok:
                    continue
                writer.write(frame)            # raw frame, without the overlay
                fw.writerow([frame_idx, f"{now:.4f}"])
                frame_idx += 1
                shown = frame.copy()
                colour = (0, 0, 255) if kind.startswith("closure") else (0, 200, 0)
                cv2.putText(shown, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            colour, 3)
                cv2.putText(shown, f"{start + dur - now:4.1f}s", (30, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
                cv2.imshow("recording", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    aborted = True
                    break
            cw.writerow([f"{start:.4f}", f"{time.perf_counter() - t0:.4f}", kind, text])
            if aborted:
                break
    finally:
        cap.release()
        writer.release()
        frames_f.close()
        cues_f.close()
        cv2.destroyAllWindows()
    dur = time.perf_counter() - t0
    print(f"Saved {frame_idx} frames in {dur:.1f} s "
          f"({frame_idx / dur:.1f} fps) to {out / session}.*"
          + ("  [ABORTED]" if aborted else ""))


if __name__ == "__main__":
    main()
