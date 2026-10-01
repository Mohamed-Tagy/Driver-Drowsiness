"""
Independent per-frame eye-closure reference for annotating recordings.

Uses the MediaPipe Face Landmarker (478 landmarks + blendshapes), which is
unrelated to the evaluated ViT model and Haar detector, to measure eyelid
closure on every frame:
    blink_l / blink_r : MediaPipe eyeBlinkLeft / eyeBlinkRight blendshapes (0-1)
    ear_l / ear_r     : Eye Aspect Ratio from the eyelid landmarks
The output is used only to propose closure intervals, which are then checked
visually (make_labels.py) before scoring.

Needs a separate environment (mediapipe pulls opencv-contrib, which clashes
with the project's opencv-python):
    python -m venv venv_ref && venv_ref\\Scripts\\pip install mediapipe
    venv_ref\\Scripts\\python video_eval/reference_eyes.py --video Videos/IMG_0566.MOV \\
        --out video_eval/recordings/P01
Model file: models/face/face_landmarker.task (MediaPipe model zoo).
"""

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

EYE_L = [33, 160, 158, 133, 153, 144]     # p1..p6 (subject's right eye, image left)
EYE_R = [362, 385, 387, 263, 373, 380]


def ear(pts, idx):
    p = pts[idx]
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / \
        (2 * np.linalg.norm(p[0] - p[3]) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--model", default="models/face/face_landmarker.task")
    ap.add_argument("--resize", default="1280x720")
    a = ap.parse_args()

    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=a.model),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1, output_face_blendshapes=True,
        min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
    lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    W, H = (int(v) for v in a.resize.split("x"))
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = lm.detect_for_video(img, int(1000 * i / fps))
        r = {"frame": i, "t": round(i / fps, 4), "face": 0, "blink_l": "", "blink_r": "",
             "ear_l": "", "ear_r": ""}
        if res.face_landmarks:
            pts = np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]])
            bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
            r.update(face=1, blink_l=round(bs["eyeBlinkRight"], 4),   # mirrored naming
                     blink_r=round(bs["eyeBlinkLeft"], 4),
                     ear_l=round(ear(pts, EYE_L), 4), ear_r=round(ear(pts, EYE_R), 4))
        rows.append(r)
        i += 1
    out = Path(a.out + "_reference.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{out}: {len(rows)} frames, face {sum(r['face'] for r in rows) / len(rows):.1%}")


if __name__ == "__main__":
    main()
