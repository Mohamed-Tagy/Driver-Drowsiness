"""
Run the prototype's decision pipeline, as described in the paper, on a
recorded session or a live camera, logging every frame and timing every stage.

Default (--detector haar) = the paper's prototype (the original demo.py):
    Haar frontal-face cascade, scaleFactor 1.1, minNeighbors 5, minSize 100x100,
    largest face; eye crops rows 0.18h-0.52h, cols 0.10w-0.45w / 0.55w-0.90w;
    both crops in one batch at 224x224 with the processor mean/std, BF16 on GPU;
    frame = closed if >= half of the valid crops are closed;
    counter +1 (max 15) on a closed frame, -1 (min 0) otherwise
    (open frame, no valid crop, or no face); Warning at 10, Critical at 15.
--detector tracker = the current demo.py face tracking (face_tracker.py:
    YuNet landmarks, levelled eye crops, Haar fallback); held frames leave
    the counter unchanged, as in demo.py.

Usage:
    # offline, every recorded frame (event-level metrics at camera rate)
    python video_eval/run_pipeline.py --video video_eval/recordings/P01_normal_....mp4

    # offline, mimicking a slower live system (keeps only frames it could process)
    python video_eval/run_pipeline.py --video ... --simulate-fps 12

    # live throughput and end-to-end latency, optionally with the Arduino
    python video_eval/run_pipeline.py --camera 0 --seconds 120 --serial COM5

Output: <video or session name>_pipeline.csv, one row per processed frame, and
<name>_pipeline.json with timing statistics.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_PATH = "models/vit-base-mrl-augmented-2/final_model"
WARNING_THRESHOLD = 10
CRITICAL_THRESHOLD = 15
DECAY_RATE = 1
IMG_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video")
    src.add_argument("--camera", type=int)
    p.add_argument("--frames-csv", help="Capture timestamps (default: <video>_frames.csv)")
    p.add_argument("--simulate-fps", type=float, default=None)
    p.add_argument("--seconds", type=float, default=120, help="Live mode duration")
    p.add_argument("--serial", default=None, help="Arduino port for live mode, e.g. COM5")
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--detector", choices=("haar", "tracker"), default="haar",
                   help="haar: the paper's prototype; tracker: current demo.py")
    p.add_argument("--resize", default=None,
                   help="WxH to scale video frames to, e.g. 1280x720 to match the "
                        "prototype's webcam resolution for phone recordings")
    p.add_argument("--out", default=None)
    return p.parse_args()


class Pipeline:
    def __init__(self, model_path, detector="haar"):
        self.tracker = None
        if detector == "tracker":
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from face_tracker import FaceTracker
            self.tracker = FaceTracker()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if self.bf16 else torch.float32
        proc = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.eval().to(self.device, dtype=dtype)
        self.mean = torch.tensor(proc.image_mean, dtype=dtype).view(3, 1, 1).to(self.device)
        self.std = torch.tensor(proc.image_std, dtype=dtype).view(3, 1, 1).to(self.device)
        self.dtype = dtype
        labels = {i: str(l).lower() for i, l in self.model.config.id2label.items()}
        closed = [i for i, l in labels.items()
                  if any(k in l for k in ("sleep", "drows", "close", "tired"))]
        self.closed_idx = closed[0] if closed else 1
        self.face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.counter = 0
        with torch.no_grad():                               # warm-up, as in demo.py
            self.model(pixel_values=torch.zeros(2, 3, IMG_SIZE, IMG_SIZE,
                                                device=self.device, dtype=dtype))
        self.sync()

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def prep(self, eye_bgr):
        rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
        r = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(r.transpose(2, 0, 1).copy()).float().div_(255.0)
        return (t.to(self.device, dtype=self.dtype) - self.mean) / self.std

    def step(self, frame):
        rec = {}
        t0 = time.perf_counter()
        hold = False
        if self.tracker is not None:
            face = self.tracker.update(frame)
            t1 = time.perf_counter()
            rec["face"] = int(face is not None)
            rec["source"] = face["source"] if face else ""
            hold = face is not None and face["source"] == "hold"
            crops = []
            if face is not None and not hold:
                le, re_, _ = self.tracker.eye_crops(frame, face)
                crops = [c for c in (le, re_) if c is not None and c.size > 0]
            faces = []
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                               minSize=(100, 100))
            t1 = time.perf_counter()
            rec["face"] = int(len(faces) > 0)
            crops = []
        if len(faces):
            x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            face = frame[y:y + h, x:x + w]
            top, bot = int(0.18 * h), int(0.52 * h)
            for x1, x2 in ((int(0.10 * w), int(0.45 * w)), (int(0.55 * w), int(0.90 * w))):
                c = face[top:bot, x1:x2]
                if c is not None and c.size > 0:
                    crops.append(c)
        t2 = time.perf_counter()
        probs = []
        if crops:
            batch = torch.stack([self.prep(c) for c in crops])
            with torch.no_grad():
                if self.bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = self.model(pixel_values=batch).logits
                else:
                    logits = self.model(pixel_values=batch).logits
                probs = torch.softmax(logits.float(), 1)[:, self.closed_idx].cpu().tolist()
        self.sync()
        t3 = time.perf_counter()
        votes = sum(p > 0.5 for p in probs)                # argmax of two classes
        closed = bool(probs) and votes >= len(probs) * 0.5
        if hold:
            pass                                           # tracker hold: unchanged
        elif closed:
            self.counter = min(self.counter + 1, CRITICAL_THRESHOLD)
        else:
            self.counter = max(self.counter - DECAY_RATE, 0)
        state = 2 if self.counter >= CRITICAL_THRESHOLD else \
            1 if self.counter >= WARNING_THRESHOLD else 0
        rec.update({"n_crops": len(crops),
                    "p_closed_left": f"{probs[0]:.4f}" if len(probs) > 0 else "",
                    "p_closed_right": f"{probs[1]:.4f}" if len(probs) > 1 else "",
                    "frame_closed": int(closed), "counter": self.counter,
                    "state": state,
                    "ms_face": round(1000 * (t1 - t0), 2),
                    "ms_crop": round(1000 * (t2 - t1), 2),
                    "ms_infer": round(1000 * (t3 - t2), 2),
                    "ms_total": round(1000 * (t3 - t0), 2)})
        return rec


def pct(values, q):
    return float(np.percentile(values, q)) if values else None


def main():
    args = parse_args()
    pipe = Pipeline(args.model, args.detector)
    rows, serial_acks = [], []

    if args.video:
        video = Path(args.video)
        name = video.with_suffix("")
        fcsv = Path(args.frames_csv) if args.frames_csv else Path(f"{name}_frames.csv")
        stamps = None
        if fcsv.exists():
            with open(fcsv, newline="") as f:
                stamps = [float(r["t"]) for r in csv.DictReader(f)]
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx = 0
        busy_until = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = stamps[idx] if stamps and idx < len(stamps) else idx / fps
            idx += 1
            if args.simulate_fps and t < busy_until:
                continue                    # a live system would still be busy
            if args.resize:
                size = tuple(int(v) for v in args.resize.split("x"))
                h0, w0 = frame.shape[:2]
                if abs(w0 / h0 - size[0] / size[1]) > 0.01:
                    raise SystemExit(f"--resize {args.resize} would distort a {w0}x{h0} video")
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            rec = pipe.step(frame)
            rec.update({"frame": idx - 1, "t": round(t, 4)})
            rows.append(rec)
            if args.simulate_fps:
                busy_until = t + 1.0 / args.simulate_fps
        cap.release()
    else:
        name = Path(args.out or f"video_eval/live_{time.strftime('%Y%m%d_%H%M%S')}")
        ser = None
        if args.serial:
            import serial
            ser = serial.Serial(args.serial, 9600, timeout=0)
            time.sleep(2.5)                 # Uno resets on connect; wait for READY
            ser.reset_input_buffer()
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        t_start, last_state, buf = time.perf_counter(), -1, b""
        while time.perf_counter() - t_start < args.seconds:
            t_grab = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                continue
            rec = pipe.step(frame)
            rec.update({"frame": len(rows), "t": round(t_grab - t_start, 4),
                        "ms_capture": round(1000 * (time.perf_counter() - t_grab)
                                            - rec["ms_total"], 2)})
            if ser and rec["state"] != last_state:
                ser.write(f"{rec['state']}\n".encode())
                rec["sent_state_at"] = round(time.perf_counter() - t_start, 4)
                last_state = rec["state"]
            if ser:
                buf += ser.read(256)
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.startswith(b"STATE:"):
                        serial_acks.append({"state": int(line[6:].strip() or -1),
                                            "t": round(time.perf_counter() - t_start, 4)})
            rows.append(rec)
        cap.release()

    out_csv = Path(args.out or name).with_name(Path(args.out or name).name + "_pipeline.csv")
    keys = sorted({k for r in rows for k in r},
                  key=lambda k: (k not in ("frame", "t"), k))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    span = rows[-1]["t"] - rows[0]["t"] if len(rows) > 1 else 0
    summary = {
        "source": args.video or f"camera {args.camera}",
        "detector": args.detector, "resize": args.resize,
        "device": str(pipe.device), "bf16": pipe.bf16,
        "gpu": torch.cuda.get_device_name(0) if pipe.device.type == "cuda" else None,
        "frames_processed": len(rows),
        "processed_fps": round((len(rows) - 1) / span, 2) if span else None,
        "simulate_fps": args.simulate_fps,
        "face_detected_fraction": round(sum(r["face"] for r in rows) / len(rows), 4),
        "latency_ms": {stage: {"mean": round(statistics.mean(v), 2),
                               "median": round(pct(v, 50), 2),
                               "p95": round(pct(v, 95), 2)}
                       for stage in ("ms_face", "ms_crop", "ms_infer", "ms_total")
                       for v in [[r[stage] for r in rows]]},
        "serial_acks": serial_acks,
    }
    out_json = out_csv.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "serial_acks"}, indent=2))
    print(f"Wrote {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
