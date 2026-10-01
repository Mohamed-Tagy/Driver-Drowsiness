"""
Record a scripted webcam session for system-level evaluation.

Standalone: needs only Python 3 and opencv-python (no model, no dataset),
so team members can run it on their own laptops.

The participant follows on-screen cues ("CLOSE EYES", "OPEN EYES",
"TURN HEAD LEFT", ...) with beeps. The script saves, in <out>/:
    <session>.mp4            the raw video (no model runs here)
    <session>_frames.csv     frame index and capture timestamp
    <session>_cues.csv       every cue interval, used as draft labels
    <session>_info.json      camera, frame rate and session details
    <session>.zip            all of the above, ready to send

Cue intervals are only *instructed* behavior. Check and correct them
against the video before scoring (see PROTOCOL.md), because people react
to cues with a delay and blink spontaneously.

Usage:
    python record_session.py                       # asks for the participant ID
    python record_session.py --participant P01 --condition normal
    python record_session.py --participant P01 --condition glasses --camera 1
"""

import argparse
import csv
import json
import platform
import random
import re
import threading
import time
import zipfile
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
# Realistic non-closure behaviours (--no-extras skips them). None of these is
# a cued closure, so any alert during them counts as a false alarm.
EXTRAS = ([("SLOW HEAVY BLINKS (like sleepy)", "slow_blinks", 30.0)]
          + [("HALF-CLOSE YOUR EYES", "half_closed", 10.0)] * 2
          + [("EYES DOWN TO YOUR LAP - keep head still", "eyes_down", 5.0)] * 3
          + [("YAWN", "yawn", 6.0)] * 3
          + [("SMILE / SQUINT", "squint", 5.0)] * 2
          + [("TALK (say anything out loud)", "talk", 30.0)]
          + [("RUB YOUR EYES", "rub_eyes", 5.0)] * 2
          + [("LEAN CLOSER to the camera", "lean_in", 10.0),
             ("LEAN BACK from the camera", "lean_back", 10.0)])
# Same detector settings as demo.py, used only for the positioning check.
FACE_MIN = 100


def build_script(seed, extras=True):
    rng = random.Random(seed)
    closures = [s for s in CLOSURE_SECONDS for _ in range(REPEATS)]
    rng.shuffle(closures)
    script = list(BASELINE)
    for s in closures:
        script.append(("OPEN EYES - look ahead", "open", rng.uniform(6.0, 10.0)))
        script.append((f"CLOSE EYES ({s:g} s) - open at the beep", f"closure_{s:g}s", s))
    script.append(("OPEN EYES - look ahead", "open", 8.0))
    for cue in HEAD_POSES:
        script.append(cue)
        script.append(("OPEN EYES - look ahead", "open", 5.0))
    if extras:
        extra = list(EXTRAS)
        rng.shuffle(extra)
        for cue in extra:
            script.append(cue)
            script.append(("OPEN EYES - look ahead", "open", 5.0))
    script.append(("LOOK AT THE ROAD - blink normally", "baseline", 60.0))
    return script


def beep(freq, ms=150):
    """Non-blocking beep (Windows); silent elsewhere."""
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()
    except ImportError:
        pass


def text(img, s, y, colour, scale=1.0, thick=2):
    cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, s, (30, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


def open_camera(index, w, h, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if platform.system() == "Windows" \
        else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None, None
    return cap, frame


def position_check(cap, cam, args):
    """Live preview until the face is well placed and SPACE is pressed.
    Returns (cap, camera index) or (None, None) if the user quit."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                    "haarcascade_frontalface_default.xml")
    t_last, n = time.perf_counter(), 0
    fps_est = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        n += 1
        now = time.perf_counter()
        if now - t_last >= 1.0:
            fps_est, n, t_last = n / (now - t_last), 0, now
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(FACE_MIN, FACE_MIN))
        shown = frame.copy()
        h, w = shown.shape[:2]
        if len(faces) == 1:
            x, y, fw, fh = faces[0]
            good = 0.20 * w <= fw <= 0.45 * w
            colour = (0, 200, 0) if good else (0, 200, 255)
            cv2.rectangle(shown, (x, y), (x + fw, y + fh), colour, 3)
            msg = "GOOD - press SPACE to start" if good else \
                ("Move CLOSER to the camera" if fw < 0.20 * w else "Move a bit FURTHER away")
        elif len(faces) == 0:
            colour, msg = (0, 0, 255), "No face found - face the camera, check the light"
        else:
            colour, msg = (0, 0, 255), "More than one face - only you in the picture"
        text(shown, msg, 50, colour)
        text(shown, f"Camera {cam}  {w}x{h}  ~{fps_est:.0f} fps", 95, (255, 255, 255), 0.7)
        text(shown, "SPACE start   C switch camera   Q quit", h - 25, (255, 255, 255), 0.7)
        if fps_est and fps_est < 20:
            text(shown, "Low frame rate: turn on more light / plug in the charger",
                 135, (0, 200, 255), 0.7)
        cv2.imshow("recording", shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None, None
        if key == ord(" "):
            args.measured_fps = fps_est
            return cap, cam
        if key == ord("c"):
            for step in range(1, 5):
                nxt = (cam + step) % 5
                new, _ = open_camera(nxt, args.width, args.height, args.fps)
                if new is not None:
                    cap.release()
                    cap, cam = new, nxt
                    break


def countdown(cap, seconds=10):
    lines = ["Keep looking at the camera like it is the road.",
             "HIGH beep: do what the text says.",
             "CLOSE EYES: keep them closed until the LOW beep.",
             "Hold each action until the next beep. About 9 minutes."]
    end = time.perf_counter() + seconds
    while (left := end - time.perf_counter()) > 0:
        ok, frame = cap.read()
        if not ok:
            continue
        for i, s in enumerate(lines):
            text(frame, s, 50 + 40 * i, (255, 255, 255), 0.8)
        text(frame, f"Starting in {left:.0f}", 240, (0, 200, 255), 1.4, 3)
        cv2.imshow("recording", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--participant", help="e.g. P01 (asked if omitted)")
    p.add_argument("--condition", default="normal",
                   help="e.g. normal, glasses, dim_light")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "recordings"))
    p.add_argument("--seed", type=int,
                   help="Order of the cues (default: the number in the participant ID)")
    p.add_argument("--no-extras", action="store_true",
                   help="Skip the yawning/talking/eyes-down/... segments")
    p.add_argument("--no-sound", action="store_true")
    args = p.parse_args()
    sound = beep if not args.no_sound else (lambda *a, **k: None)

    if not args.participant:
        args.participant = input("Participant ID (e.g. P01): ").strip() or "P00"
        c = input("Condition [normal / glasses / dim_light] (Enter = normal): ").strip()
        args.condition = c or args.condition
    args.participant = re.sub(r"[^A-Za-z0-9]", "", args.participant).upper()
    if args.seed is None:
        digits = re.sub(r"\D", "", args.participant)
        args.seed = int(digits) if digits else 0
        if args.condition != "normal":
            args.seed += 100

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap, frame = open_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        raise SystemExit(f"Cannot read from camera {args.camera}. "
                         "Close other apps using the camera (Zoom, Teams, browser) "
                         "or try --camera 1.")
    cap, cam = position_check(cap, args.camera, args)
    if cap is None:
        cv2.destroyAllWindows()
        raise SystemExit("Quit before recording.")
    if not countdown(cap):
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit("Quit before recording.")

    ok, frame = cap.read()
    h, w = frame.shape[:2]
    session = f"{args.participant}_{args.condition}_{time.strftime('%Y%m%d_%H%M%S')}"
    # Write at the rate the camera actually delivers (measured in the preview),
    # so that video players show the same timeline as _frames.csv.
    fps_out = getattr(args, "measured_fps", 0.0) or args.fps
    fps_out = round(fps_out) if fps_out > 5 else args.fps
    writer = cv2.VideoWriter(str(out / f"{session}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (w, h))

    script = build_script(args.seed, extras=not args.no_extras)
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
        for i, (cue, kind, dur) in enumerate(script):
            start = time.perf_counter() - t0
            nxt = script[i + 1] if i + 1 < len(script) else None
            # High beep = close now; low beep = open now (eyes are shut, so
            # the end of a closure must be audible). Mid beep for other cues.
            sound(1200 if kind.startswith("closure") else 600 if kind == "open" else 900)
            while time.perf_counter() - t0 < start + dur:
                ok, frame = cap.read()
                now = time.perf_counter() - t0      # stamp after the read
                if not ok:
                    continue
                writer.write(frame)            # raw frame, without the overlay
                fw.writerow([frame_idx, f"{now:.4f}"])
                frame_idx += 1
                shown = frame.copy()
                colour = (0, 0, 255) if kind.startswith("closure") else (0, 200, 0)
                text(shown, cue, 60, colour, 1.2, 3)
                text(shown, f"{start + dur - now:4.1f}s", 110, colour)
                if nxt and start + dur - now < 3.0:
                    text(shown, f"NEXT: {nxt[0]}", 160, (0, 200, 255), 0.8)
                text(shown, f"{now / 60:.1f} / {total / 60:.1f} min", h - 25,
                     (255, 255, 255), 0.7)
                cv2.imshow("recording", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    aborted = True
                    break
            cw.writerow([f"{start:.4f}", f"{time.perf_counter() - t0:.4f}", kind, cue])
            if aborted:
                break
    finally:
        cap.release()
        writer.release()
        frames_f.close()
        cues_f.close()
        cv2.destroyAllWindows()
    dur = time.perf_counter() - t0
    fps = frame_idx / dur if dur else 0.0

    info = {"session": session, "participant": args.participant,
            "condition": args.condition, "seed": args.seed,
            "extras": not args.no_extras, "aborted": aborted,
            "camera_index": cam, "resolution": [w, h],
            "requested_fps": args.fps, "achieved_fps": round(fps, 2),
            "frames": frame_idx, "duration_s": round(dur, 1),
            "platform": platform.platform(), "python": platform.python_version(),
            "opencv": cv2.__version__}
    (out / f"{session}_info.json").write_text(json.dumps(info, indent=2))

    zpath = out / f"{session}.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.write(out / f"{session}.mp4", f"{session}.mp4", zipfile.ZIP_STORED)
        for suffix in ("_frames.csv", "_cues.csv", "_info.json"):
            z.write(out / f"{session}{suffix}", f"{session}{suffix}", zipfile.ZIP_DEFLATED)

    print(f"\nSaved {frame_idx} frames in {dur:.1f} s ({fps:.1f} fps).")
    if aborted:
        print("ABORTED - please record again.")
    elif fps < 20:
        print("WARNING: low frame rate. If possible, record again with more light "
              "and the charger plugged in.")
    print(f"\nSEND THIS FILE: {zpath.resolve()}  ({zpath.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
