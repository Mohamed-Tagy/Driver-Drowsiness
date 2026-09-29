"""
Drowsiness Detection Demo
Model: ViT-Base + MRL Augmented 2
Fixed: FPS drop, batched inference, BF16, no model switching
"""

import cv2
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)
import time
import os
import sys
import numpy as np

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("pyserial not installed - Arduino disabled")


# =============================================================
# CONFIGURATION
# =============================================================

MODEL_PATH   = "models/vit-base-mrl-augmented-2/final_model"
MODEL_NAME   = "ViT-Base + MRL Augmented 2"

CAMERA_INDEX             = 0
MAX_RETRIES              = 30
WARNING_THRESHOLD        = 10
CRITICAL_THRESHOLD       = 15
DECAY_RATE               = 1
ARDUINO_PORT             = None  #None for auto-detect
ARDUINO_BAUD             = 9600
INFERENCE_EVERY_N_FRAMES = 1
IMG_SIZE                 = 224


# =============================================================
# LOAD MODEL
# =============================================================

print("=" * 60)
print("  DROWSINESS DETECTION DEMO")
print(f"  Model: {MODEL_NAME}")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    print(f"\nERROR: Model not found at: {MODEL_PATH}")
    print("Make sure the model is trained and saved.")
    sys.exit(1)

print(f"\nLoading model from: {MODEL_PATH}")

image_processor = AutoImageProcessor.from_pretrained(
    MODEL_PATH)
model = AutoModelForImageClassification.from_pretrained(
    MODEL_PATH)
model.eval()

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')

# BF16 for RTX 5050 - cuts inference time in half
USE_BF16 = (
    device.type == 'cuda' and
    torch.cuda.is_bf16_supported())

if USE_BF16:
    model = model.to(torch.bfloat16)
    print("BF16 inference enabled")

model = model.to(device)

print(f"Device:  {device}")
print(f"BF16:    {USE_BF16}")
print(f"Classes: {model.config.id2label}")

# Pre-compute normalization constants on GPU
# Avoids recomputing them every frame
IMG_MEAN = torch.tensor(
    image_processor.image_mean,
    dtype=torch.bfloat16 if USE_BF16
    else torch.float32
).view(3, 1, 1).to(device)

IMG_STD = torch.tensor(
    image_processor.image_std,
    dtype=torch.bfloat16 if USE_BF16
    else torch.float32
).view(3, 1, 1).to(device)

# Detect labels automatically
labels       = model.config.id2label
drowsy_label = None
awake_label  = None

for idx, label in labels.items():
    ll = str(label).lower()
    if any(k in ll for k in
           ['sleep', 'drows', 'close', 'tired']):
        drowsy_label = label
    elif any(k in ll for k in
             ['awake', 'alert', 'open']):
        awake_label = label

if drowsy_label is None or awake_label is None:
    label_list = list(labels.values())
    if len(label_list) >= 2:
        awake_label  = label_list[0]
        drowsy_label = label_list[1]
    else:
        print("ERROR: Could not determine labels.")
        sys.exit(1)

print(f"Awake:   '{awake_label}'")
print(f"Drowsy:  '{drowsy_label}'")


# =============================================================
# FAST PREPROCESSING
# =============================================================

def preprocess_eye(eye_bgr):
    """
    Fast OpenCV + PyTorch preprocessing.
    No PIL overhead. Runs on GPU.

    BGR → RGB → resize 224x224
    → normalize with ImageNet stats
    → tensor on GPU
    """
    rgb     = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb, (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(
        resized.transpose(2, 0, 1).copy()
    ).float().div_(255.0)

    tensor = tensor.to(
        device,
        dtype=torch.bfloat16 if USE_BF16
        else torch.float32,
        non_blocking=True)

    tensor = (tensor - IMG_MEAN) / IMG_STD
    return tensor   # [3, 224, 224]


# =============================================================
# BATCHED EYE PREDICTION
# =============================================================

def predict_eyes_batch(left_bgr, right_bgr):
    """
    Both eyes in ONE forward pass.
    Halves GPU overhead vs two separate calls.
    """
    tensors = []
    sides   = []

    if left_bgr is not None and left_bgr.size > 0:
        try:
            tensors.append(preprocess_eye(left_bgr))
            sides.append('left')
        except Exception:
            pass

    if right_bgr is not None and right_bgr.size > 0:
        try:
            tensors.append(preprocess_eye(right_bgr))
            sides.append('right')
        except Exception:
            pass

    if not tensors:
        return False, False, 0.0, 0.0, None, None

    # Stack: [N, 3, 224, 224]
    batch = torch.stack(tensors, dim=0)

    with torch.no_grad():
        if USE_BF16:
            with torch.autocast(
                    device_type='cuda',
                    dtype=torch.bfloat16):
                outputs = model(pixel_values=batch)
        else:
            outputs = model(pixel_values=batch)

        probs = torch.softmax(
            outputs.logits.float(), dim=1)

    left_drowsy  = False
    right_drowsy = False
    left_conf    = 0.0
    right_conf   = 0.0
    left_probs   = None
    right_probs  = None

    for i, side in enumerate(sides):
        p         = probs[i].cpu().numpy()
        pred_idx  = int(probs[i].argmax().item())
        conf      = float(probs[i][pred_idx].item())
        label     = model.config.id2label[pred_idx]
        is_drowsy = (str(label).lower() ==
                     str(drowsy_label).lower())

        if side == 'left':
            left_drowsy = is_drowsy
            left_conf   = conf
            left_probs  = p
        else:
            right_drowsy = is_drowsy
            right_conf   = conf
            right_probs  = p

    return (left_drowsy, right_drowsy,
            left_conf, right_conf,
            left_probs, right_probs)


# =============================================================
# ARDUINO
# =============================================================

ser            = None
serial_enabled = False
arduino_port   = None


def list_all_ports():
    if not SERIAL_AVAILABLE:
        return
    ports = list(serial.tools.list_ports.comports())
    print("\n--- Available COM Ports ---")
    if not ports:
        print("  No COM ports found!")
    else:
        for p in ports:
            print(f"  {p.device:8s} | {p.description}")
    print("---------------------------\n")


def try_open_serial():
    global arduino_port

    if not SERIAL_AVAILABLE:
        return None

    list_all_ports()

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No COM ports found.")
        return None

    if ARDUINO_PORT is not None:
        print(f"Trying: {ARDUINO_PORT}")
        for attempt in range(1, 6):
            try:
                s = serial.Serial(
                    ARDUINO_PORT, ARDUINO_BAUD,
                    timeout=2)
                time.sleep(2)
                s.flushInput()
                arduino_port = ARDUINO_PORT
                print(f"Arduino connected: {ARDUINO_PORT}")
                return s
            except PermissionError:
                print(
                    f"  Attempt {attempt}/5: "
                    f"Busy, waiting 3s...")
                time.sleep(3)
            except Exception as e:
                print(f"  Attempt {attempt}/5: {e}")
                time.sleep(2)
        print("Failed after 5 attempts.")
        return None

    # Auto-detect fallback
    def port_priority(p):
        d = p.description.lower()
        if 'arduino'      in d: return 0
        elif 'ch340'      in d: return 1
        elif 'ch341'      in d: return 2
        elif 'usb serial' in d: return 3
        else: return 9

    for port in sorted(ports, key=port_priority):
        print(
            f"Trying {port.device} "
            f"({port.description})... ",
            end="", flush=True)
        for attempt in range(1, 4):
            try:
                s = serial.Serial(
                    port.device, ARDUINO_BAUD, timeout=2)
                time.sleep(2)
                s.flushInput()
                s.write(b"0\n")
                time.sleep(0.1)
                arduino_port = port.device
                print("Connected!")
                return s
            except PermissionError:
                if attempt < 3:
                    print(
                        f"\n  Busy, retry {attempt}/3...",
                        end="", flush=True)
                    time.sleep(3)
                else:
                    print("\n  Permission denied")
            except Exception as e:
                print(f"Failed ({str(e)[:40]})")
                break
    return None


def reconnect_arduino():
    global ser, serial_enabled, arduino_port
    if arduino_port:
        try:
            s = serial.Serial(
                arduino_port, ARDUINO_BAUD, timeout=2)
            time.sleep(2)
            s.flushInput()
            ser            = s
            serial_enabled = True
            print(f"Reconnected: {arduino_port}")
            return True
        except Exception:
            pass
    s = try_open_serial()
    if s:
        ser            = s
        serial_enabled = True
        return True
    return False


def send_to_arduino(state):
    global ser, serial_enabled
    if not serial_enabled or ser is None:
        return
    try:
        ser.write(f"{state}\n".encode())
    except Exception as e:
        print(f"\nArduino disconnected: {e}")
        serial_enabled = False
        try:
            ser.close()
        except Exception:
            pass
        ser = None


ser = try_open_serial()
if ser:
    serial_enabled = True


# =============================================================
# FACE DETECTOR
# =============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml")


# =============================================================
# CAMERA
# =============================================================

print(f"\nOpening camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: No camera found!")
    sys.exit(1)

print(f"Camera {CAMERA_INDEX} opened!")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# =============================================================
# GPU WARMUP
# =============================================================

print("\nWarming up GPU...")
with torch.no_grad():
    _dummy = torch.zeros(
        2, 3, IMG_SIZE, IMG_SIZE,
        device=device,
        dtype=torch.bfloat16 if USE_BF16
        else torch.float32)
    _ = model(pixel_values=_dummy)
    del _dummy
if device.type == 'cuda':
    torch.cuda.synchronize()
print("GPU ready\n")


# =============================================================
# STATE
# =============================================================

sleepy_counter   = 0
alert_state      = 0
last_sent_state  = -1
frame_count      = 0

fps_time         = time.time()
fps_count        = 0
current_fps      = 0.0
retry_count      = 0

last_reconnect        = 0
RECONNECT_INTERVAL    = 10

# Cached prediction shown on skipped frames
last_final_label     = ""
last_color           = (0, 255, 0)
last_avg_conf        = 0.0
last_avg_awake_prob  = 0.0
last_avg_drowsy_prob = 0.0

print("=" * 60)
print("DEMO STARTED")
print("=" * 60)
print("Controls:")
print("  Q = Quit")
print("  S = Screenshot")
print("  C = Cycle camera")
print("  R = Reset counters")
print("  A = Retry Arduino connection")
print("  F = Toggle frame skip")
print()
print(f"Warning threshold:  {WARNING_THRESHOLD} frames")
print(f"Critical threshold: {CRITICAL_THRESHOLD} frames")
print(f"Arduino:  "
      f"{'Connected on ' + str(arduino_port) if serial_enabled else 'Not connected'}")
print()


# =============================================================
# MAIN LOOP
# =============================================================

os.makedirs("screenshots", exist_ok=True)

while True:
    ret, frame = cap.read()

    if not ret:
        retry_count += 1
        if retry_count >= MAX_RETRIES:
            print(f"\nCamera {CAMERA_INDEX} stopped!")
            retry_count = 0
            time.sleep(1)
        else:
            if retry_count % 10 == 1:
                print(f"No frame "
                      f"({retry_count}/{MAX_RETRIES})")
            time.sleep(0.1)
        continue

    retry_count  = 0
    frame_count += 1

    # FPS
    fps_count += 1
    elapsed = time.time() - fps_time
    if elapsed >= 1.0:
        current_fps = fps_count / elapsed
        fps_count   = 0
        fps_time    = time.time()

    # Auto-reconnect Arduino
    if (not serial_enabled and
            time.time() - last_reconnect >
            RECONNECT_INTERVAL):
        last_reconnect = time.time()
        reconnect_arduino()

    run_inference = (
        frame_count % INFERENCE_EVERY_N_FRAMES == 0)

    # ── Face detection ────────────────────────────────────
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100))

    if len(faces) > 0:
        faces = sorted(
            faces,
            key=lambda f: f[2] * f[3],
            reverse=True)
        (x, y, w, h) = faces[0]

        face_color = frame[y:y+h, x:x+w]

        eye_top  = int(0.18 * h)
        eye_bot  = int(0.52 * h)
        left_x1  = int(0.10 * w)
        left_x2  = int(0.45 * w)
        right_x1 = int(0.55 * w)
        right_x2 = int(0.90 * w)

        left_eye  = face_color[
            eye_top:eye_bot, left_x1:left_x2]
        right_eye = face_color[
            eye_top:eye_bot, right_x1:right_x2]

        cv2.rectangle(
            frame, (x, y), (x+w, y+h),
            (0, 255, 0), 2)
        cv2.rectangle(
            frame,
            (x+left_x1,  y+eye_top),
            (x+left_x2,  y+eye_bot),
            (255, 200, 0), 2)
        cv2.rectangle(
            frame,
            (x+right_x1, y+eye_top),
            (x+right_x2, y+eye_bot),
            (255, 200, 0), 2)

        if run_inference:
            (left_drowsy, right_drowsy,
             left_conf,   right_conf,
             left_probs,  right_probs
             ) = predict_eyes_batch(left_eye, right_eye)

            drowsy_votes = sum([left_drowsy, right_drowsy])
            total_votes  = sum([
                left_probs  is not None,
                right_probs is not None])

            if total_votes > 0:
                all_probs = [
                    p for p in [left_probs, right_probs]
                    if p is not None]
                avg_probs = np.mean(all_probs, axis=0)

                last_avg_awake_prob  = float(avg_probs[0])
                last_avg_drowsy_prob = float(avg_probs[1])
                last_avg_conf        = max(
                    left_conf, right_conf)

                if drowsy_votes >= total_votes * 0.5:
                    sleepy_counter = min(
                        sleepy_counter + 1,
                        CRITICAL_THRESHOLD)
                    last_final_label = drowsy_label.upper()
                    last_color       = (0, 0, 255)
                else:
                    sleepy_counter = max(
                        sleepy_counter - DECAY_RATE, 0)
                    last_final_label = awake_label.upper()
                    last_color       = (0, 255, 0)
            else:
                sleepy_counter = max(
                    sleepy_counter - DECAY_RATE, 0)

        # Draw cached label
        if last_final_label:
            cv2.putText(
                frame,
                f"{last_final_label}: "
                f"{last_avg_conf:.0%}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, last_color, 2)
            cv2.putText(
                frame,
                f"Awake:  {last_avg_awake_prob:.1%}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Sleepy: {last_avg_drowsy_prob:.1%}",
                (x, y + h + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)

    else:
        sleepy_counter = max(
            sleepy_counter - DECAY_RATE, 0)
        cv2.putText(
            frame, "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2)

    # ── Alert state ──────────────────────────────────────
    if sleepy_counter >= CRITICAL_THRESHOLD:
        alert_state = 2
    elif sleepy_counter >= WARNING_THRESHOLD:
        alert_state = 1
    else:
        alert_state = 0

    # ── Send to Arduino ──────────────────────────────────
    if alert_state != last_sent_state:
        send_to_arduino(alert_state)
        last_sent_state = alert_state

    # ── Status panel ─────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(
        overlay, (10, 50), (430, 300),
        (0, 0, 0), -1)
    cv2.addWeighted(
        overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    infer_rate = current_fps / max(
        INFERENCE_EVERY_N_FRAMES, 1)
    cv2.putText(
        frame,
        f"FPS: {current_fps:.1f}  "
        f"Infer: {infer_rate:.1f}/s",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 0), 2)

    # Counter
    counter_color = (
        (0, 255, 0)
        if sleepy_counter < WARNING_THRESHOLD
        else (0, 165, 255)
        if sleepy_counter < CRITICAL_THRESHOLD
        else (0, 0, 255))

    cv2.putText(
        frame,
        f"Counter: {sleepy_counter}/{CRITICAL_THRESHOLD}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, counter_color, 2)

    # Progress bar
    bx, by, bw, bh = 20, 113, 390, 10
    cv2.rectangle(
        frame, (bx, by), (bx+bw, by+bh),
        (60, 60, 60), -1)

    warn_x = bx + int(
        WARNING_THRESHOLD / CRITICAL_THRESHOLD * bw)
    cv2.rectangle(
        frame, (warn_x-1, by), (warn_x+1, by+bh),
        (0, 165, 255), -1)
    cv2.putText(
        frame, "W",
        (warn_x - 5, by - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 165, 255), 1)

    bar_fill = int(
        min(sleepy_counter / CRITICAL_THRESHOLD, 1.0)
        * bw)
    if bar_fill > 0:
        cv2.rectangle(
            frame, (bx, by), (bx+bar_fill, by+bh),
            counter_color, -1)

    # State
    state_names  = ["NORMAL", "WARNING", "CRITICAL"]
    state_colors = [
        (0, 255, 0), (0, 165, 255), (0, 0, 255)]

    cv2.putText(
        frame,
        f"State: {state_names[alert_state]}",
        (20, 143),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, state_colors[alert_state], 2)

    # Time to next state
    if alert_state == 0 and sleepy_counter > 0:
        fl   = WARNING_THRESHOLD - sleepy_counter
        secs = fl / max(current_fps, 1)
        cv2.putText(
            frame,
            f"-> Warning in ~{secs:.1f}s",
            (185, 143),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (180, 180, 180), 1)
    elif alert_state == 1:
        fl   = CRITICAL_THRESHOLD - sleepy_counter
        secs = fl / max(current_fps, 1)
        cv2.putText(
            frame,
            f"-> Critical in ~{secs:.1f}s",
            (185, 143),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (0, 165, 255), 1)

    # Arduino
    ard_txt   = (f"Arduino: {arduino_port}"
                 if serial_enabled
                 else "Arduino: NOT CONNECTED (A=retry)")
    ard_color = (
        (0, 255, 0) if serial_enabled else (0, 0, 255))
    cv2.putText(
        frame, ard_txt,
        (20, 172),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58, ard_color, 2)

    # Camera
    cv2.putText(
        frame, f"Camera: {CAMERA_INDEX}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2)

    # Model name (fixed - no switching)
    cv2.putText(
        frame, f"Model: {MODEL_NAME}",
        (20, 228),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (200, 255, 200), 2)

    # Controls hint
    cv2.putText(
        frame,
        "Q=Quit  S=Screenshot  C=Camera  "
        "R=Reset  A=Arduino  F=Skip",
        (20, 255),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40, (150, 150, 150), 1)

    # ── Alert banner ─────────────────────────────────────
    h_frame, w_frame = frame.shape[:2]

    if alert_state == 1:
        cv2.rectangle(
            frame,
            (0, h_frame - 50),
            (w_frame, h_frame),
            (0, 165, 255), -1)
        cv2.putText(
            frame,
            "WARNING: POSSIBLE DROWSINESS",
            (w_frame // 2 - 250, h_frame - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 2)

    elif alert_state == 2:
        flash = int(time.time() * 3) % 2
        cv2.rectangle(
            frame,
            (0, h_frame - 50),
            (w_frame, h_frame),
            (0, 0, 255) if flash else (0, 0, 180), -1)
        cv2.putText(
            frame,
            "CRITICAL: WAKE UP!",
            (w_frame // 2 - 200, h_frame - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (255, 255, 255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nQuitting...")
        break

    elif key == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"screenshots/screenshot_{ts}.jpg"
        cv2.imwrite(fn, frame)
        print(f"Screenshot: {fn}")

    elif key == ord('r'):
        sleepy_counter  = 0
        alert_state     = 0
        last_sent_state = -1
        print("Counters reset")

    elif key == ord('f'):
        INFERENCE_EVERY_N_FRAMES = (
            2 if INFERENCE_EVERY_N_FRAMES == 1 else 1)
        print(
            f"Frame skip: {INFERENCE_EVERY_N_FRAMES} "
            f"({'faster' if INFERENCE_EVERY_N_FRAMES > 1 else 'accurate'})")

    elif key == ord('a'):
        print("\nManual Arduino reconnect...")
        if ser:
            try:
                ser.close()
            except Exception:
                pass
            ser = None
        serial_enabled = False
        last_reconnect = 0
        if reconnect_arduino():
            print(f"Reconnected: {arduino_port}")
        else:
            print("Still not connected")

    elif key == ord('c'):
        print(f"\nSwitching camera...")
        old_cam = CAMERA_INDEX
        cap.release()
        found = False
        for attempt in range(5):
            CAMERA_INDEX = (old_cam + attempt + 1) % 5
            print(f"Trying {CAMERA_INDEX}...", end=" ")
            cap = cv2.VideoCapture(
                CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(CAMERA_INDEX)
            if cap.isOpened():
                ret2, _ = cap.read()
                if ret2:
                    cap.set(
                        cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(
                        cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    print("OK!")
                    found = True
                    break
                else:
                    cap.release()
                    print("No data")
            else:
                print("N/A")
        if not found:
            CAMERA_INDEX = old_cam
            cap = cv2.VideoCapture(
                CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"Reverted to {CAMERA_INDEX}")
        retry_count = 0


# =============================================================
# CLEANUP
# =============================================================

cap.release()
if ser:
    try:
        ser.write(b"0\n")
        ser.close()
    except Exception:
        pass
cv2.destroyAllWindows()
print("Demo ended")