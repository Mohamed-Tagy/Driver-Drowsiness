"""
Face and eye tracking for demo.py.

Replaces the single Haar-cascade call, which loses the face as soon as the
head tilts or turns. Order of attempts on every frame:
    1. YuNet CNN face detector (OpenCV, models/face/...onnx), which also
       returns both eye centres and copes with head roll, yaw and pitch;
    2. YuNet on the frame rotated by +/-30 deg (strong head tilt);
    3. Haar frontal cascade (the original detector) as a last resort;
    4. the last known position, held for a few frames (brief misses).
Eye crops are cut around the eye landmarks after rotating the face level,
with the same size and offset as the original fixed-ratio Haar crops on a
frontal face (calibrated on the evaluation videos: crop 1.05 x 1.02 times
the inter-eye distance, centred 0.18 outward and 0.13 above each eye).
"""

import os

import cv2
import numpy as np

YUNET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "face", "face_detection_yunet_2023mar.onnx")

CROP_W, CROP_H = 1.05, 1.02      # crop size / inter-eye distance
OFF_X, OFF_Y = 0.18, -0.13       # crop centre offset from the eye / IED
DET_WIDTH = 640                  # detection runs on a downscaled frame
ROTATIONS = (30, -30)            # fallback angles for strong tilt
REFINE_ROLL = 20                 # re-detect on a levelled frame above this roll
HOLD_FRAMES = 8                  # keep the last face this long when lost
SMOOTH = 0.5                     # landmark smoothing (0 = none)


class FaceTracker:
    def __init__(self, yunet_path=YUNET_PATH, score=0.6):
        self.yunet = None
        if os.path.exists(yunet_path):
            self.yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), score)
        else:
            print(f"YuNet model not found at {yunet_path}; using Haar only")
        self.haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.last = None          # (box, eye_l, eye_r) in full-frame pixels
        self.missed = 0           # frames held since the last detection
        self.no_face = 0          # consecutive frames without a detection

    # -- detectors --------------------------------------------------------
    def _yunet(self, img):
        """Faces as (box, eye_l, eye_r, score); eye_l is the image-left eye."""
        h, w = img.shape[:2]
        s = min(1.0, DET_WIDTH / w)
        small = cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img
        self.yunet.setInputSize((small.shape[1], small.shape[0]))
        _, dets = self.yunet.detect(small)
        out = []
        for d in (dets if dets is not None else []):
            d = d / s
            e1, e2 = d[4:6], d[6:8]
            eye_l, eye_r = (e1, e2) if e1[0] <= e2[0] else (e2, e1)
            out.append((d[:4], eye_l, eye_r, d[14] * s))
        return out

    def _yunet_rotated(self, frame, angle):
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(frame, M, (w, h))
        Minv = cv2.invertAffineTransform(M)
        back = lambda p: Minv[:, :2] @ p + Minv[:, 2]
        out = []
        for box, el, er, sc in self._yunet(rot):
            x, y, bw, bh = box
            corners = np.array([back(np.array(c)) for c in
                                [(x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)]])
            x0, y0 = corners.min(0)
            x1, y1 = corners.max(0)
            a, b = back(el), back(er)
            eye_l, eye_r = (a, b) if a[0] <= b[0] else (b, a)
            out.append((np.array([x0, y0, x1 - x0, y1 - y0]), eye_l, eye_r, sc))
        return out

    def _haar(self, frame):
        # Runs on the downscaled frame (full resolution costs ~25 ms at 1080p)
        h0, w0 = frame.shape[:2]
        s = min(1.0, DET_WIDTH / w0)
        small = cv2.resize(frame, (int(w0 * s), int(h0 * s))) if s < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        min_side = max(40, int(100 * s))
        faces = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(min_side, min_side))
        out = []
        for x, y, w, h in (np.asarray(f, float) / s for f in faces):
            ied = w / 3.0          # measured face width / IED on frontal faces
            eye_l = np.array([x + 0.275 * w - OFF_X * -ied, y + 0.35 * h - OFF_Y * ied])
            eye_r = np.array([x + 0.725 * w - OFF_X * ied, y + 0.35 * h - OFF_Y * ied])
            out.append((np.array([x, y, w, h], float), eye_l, eye_r, 0.0))
        return out

    # -- tracking ----------------------------------------------------------
    def _pick(self, cands):
        if not cands:
            return None
        if self.last is None:
            return max(cands, key=lambda c: (c[3], c[0][2] * c[0][3]))
        pc = (self.last[1] + self.last[2]) / 2
        return min(cands, key=lambda c: np.linalg.norm((c[1] + c[2]) / 2 - pc))

    def update(self, frame):
        """Returns dict(box, eye_l, eye_r, source) or None if no face."""
        cand, source = [], None
        if self.yunet is not None:
            cand, source = self._yunet(frame), "yunet"
            # Rotated retry (both directions; keep the more confident one,
            # since the wrong direction can still "find" a distorted face).
            # Skipped on every other missed frame to bound the cost of
            # frames without a face.
            if not cand and self.no_face % 2 == 0:
                tries = [self._yunet_rotated(frame, a) for a in ROTATIONS]
                best_try = max(tries, key=lambda c: max((x[3] for x in c), default=-1.0))
                if best_try:
                    cand, source = best_try, "yunet-rot"
        if not cand:
            cand, source = self._haar(frame), "haar"
        best = self._pick(cand)

        if best is None:
            self.no_face += 1
            if self.last is not None and self.missed < HOLD_FRAMES:
                self.missed += 1
                box, el, er = self.last
                return dict(box=box, eye_l=el, eye_r=er, source="hold")
            self.last, self.missed = None, 0
            return None

        self.no_face = 0
        box, el, er, _ = best
        # Strong head roll: YuNet landmarks degrade (~20-30% IED error at
        # 45 deg), so re-detect on the frame rotated to level the eyes.
        if self.yunet is not None:
            roll = np.degrees(np.arctan2(er[1] - el[1], er[0] - el[0]))
            if abs(roll) > REFINE_ROLL:
                refined = self._yunet_rotated(frame, roll)
                if refined:
                    mid = (el + er) / 2
                    rb = min(refined, key=lambda c: np.linalg.norm((c[1] + c[2]) / 2 - mid))
                    box, el, er = rb[0], rb[1], rb[2]
        if self.last is not None and SMOOTH > 0:
            ied = np.linalg.norm(er - el)
            if np.linalg.norm((el + er) / 2 - (self.last[1] + self.last[2]) / 2) < 0.3 * ied:
                el = SMOOTH * self.last[1] + (1 - SMOOTH) * el
                er = SMOOTH * self.last[2] + (1 - SMOOTH) * er
        self.last, self.missed = (np.asarray(box, float), el, er), 0
        return dict(box=self.last[0], eye_l=el, eye_r=er, source=source)

    # -- eye crops ---------------------------------------------------------
    @staticmethod
    def eye_crops(frame, face):
        """Level the eye line and cut both eye crops.
        Returns (left_crop, right_crop, [quad_left, quad_right]) where the
        quads are the crop corners in frame pixels, for drawing."""
        el, er = face["eye_l"], face["eye_r"]
        d = er - el
        ied = float(np.hypot(*d))
        if ied < 8:
            return None, None, []
        angle = np.degrees(np.arctan2(d[1], d[0]))
        mid = (el + er) / 2
        M = cv2.getRotationMatrix2D((float(mid[0]), float(mid[1])), angle, 1.0)
        Minv = cv2.invertAffineTransform(M)
        cw, ch = CROP_W * ied, CROP_H * ied
        crops, quads = [], []
        for eye, sign in ((el, -1), (er, 1)):
            c = M[:, :2] @ eye + M[:, 2] + np.array([sign * OFF_X * ied, OFF_Y * ied])
            x0, y0 = c[0] - cw / 2, c[1] - ch / 2
            T = M.copy()
            T[:, 2] -= (x0, y0)
            crop = cv2.warpAffine(frame, T, (max(1, int(round(cw))), max(1, int(round(ch)))),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            crops.append(crop)
            corners = [(x0, y0), (x0 + cw, y0), (x0 + cw, y0 + ch), (x0, y0 + ch)]
            quads.append(np.array([Minv[:, :2] @ np.array(p) + Minv[:, 2] for p in corners],
                                  np.int32))
        return crops[0], crops[1], quads
