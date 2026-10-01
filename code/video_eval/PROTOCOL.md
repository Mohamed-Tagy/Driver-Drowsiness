# System-level evaluation protocol

Purpose: measure what the instructor asked for (comments #1, #4, #16, #23, #28):
- event-level detection of sustained eye closures;
- alert latency from the onset of eye closure;
- false alarms per hour;
- face-detection failures by head pose;
- end-to-end frame rate and latency.

None of this needs model training. It uses the selected checkpoint
`models/vit-base-mrl-augmented-2/final_model` and the prototype logic described
in the paper (Haar face detection, fixed-ratio crops; `run_pipeline.py
--detector haar`, the default). The current `demo.py` uses the improved face
tracker in `face_tracker.py`; evaluate it with `--detector tracker`.

> **What was actually done for the paper (Oct 2026).** The six sessions were
> recorded on a phone (1920×1080, 30 fps) with cues spoken by an experimenter
> instead of `record_session.py`, so there are no `_cues.csv`/`_frames.csv`
> files. The workflow in section 3b was used instead of manual annotation.

## 1. Participants and conditions

- **Participants:** at least 5, ideally 8–10, with varied sex, eye shape and skin tone. MRL images are infrared and come from different people, so all participants are unseen by the model.
- **Sessions per participant:**
  - `normal`: room lighting, no glasses (required);
  - `glasses`: clear glasses (if available);
  - `dim_light`: room lights dimmed (optional).
- **Consent:** get written consent to record. Do not publish identifiable video without it. The scores do not need the video to be shared.
- **Hardware:** use the webcam, laptop and GPU from the prototype description, at 1280×720, sitting at a normal driving-to-screen distance (about 60–80 cm).

## 2. Recording

```
python video_eval/record_session.py --participant P01 --condition normal --seed 1
```

The script runs for about 9 minutes. It shows cues, records the raw video (without the cue overlay), and saves capture timestamps and cue intervals. The cue order is randomized per `--seed`; use a different seed for each participant.

**Script (per session):**
1. 60 s of normal driving gaze with natural blinking.
2. 15 instructed eye closures (0.5, 1, 2, 3 and 5 s, three of each, random order), separated by 6–10 s of eyes open.
3. Head poses held for 5 s each: about 30° left and right, about 20° down and up.
4. Realistic non-closure behaviours in random order (skip with `--no-extras`): 30 s slow heavy blinks, half-closed eyes (2 × 10 s), eyes down to the lap with the head still (3 × 5 s), yawning (3 × 6 s), smiling/squinting (2 × 5 s), 30 s talking, rubbing the eyes (2 × 5 s), leaning closer and back (10 s each).
5. 60 s of normal driving gaze.

A high beep means "close now", a low beep means "open now" (the participant cannot see the screen with closed eyes), and the next cue is previewed 3 s ahead. The full session takes about 9 minutes.

The participant should keep looking toward the road (the camera area) except during head-pose cues.

## 3. Annotation (required before reporting)

`<session>_cues.csv` records what was *instructed*, not what happened: people close their eyes a few hundred milliseconds after the cue and blink spontaneously. For each session, create `<session>_labels.csv` with the same columns (`start_s,end_s,type`):

1. Copy `<session>_cues.csv` to `<session>_labels.csv`.
2. Open the video in a player that shows the time, e.g. VLC with the time display in milliseconds, or step through frames and use `<session>_frames.csv` to convert frame numbers to seconds.
3. For every `closure_*` row, set `start_s` to the first frame with the eyelids fully covering the pupil and `end_s` to the first frame they open again. Keep the `type`.
4. Add a row `start_s,end_s,blink` for every spontaneous blink or closure outside the cued closures. Blinks are reported separately and are *not* counted as misses.
5. Ideally, a second person annotates two sessions independently; report the mean absolute difference in onset times.

The scorer uses `_labels.csv` when it exists, and otherwise falls back to `_cues.csv`, flagging the results as DRAFT.

## 3b. Semi-automatic annotation (used for the paper)

1. **Reference eye closure** with an independent model (MediaPipe Face
   Landmarker), in a separate environment because `mediapipe` installs
   `opencv-contrib-python`, which clashes with the project's `opencv-python`:

   ```
   python -m venv venv_ref
   venv_ref\Scripts\pip install -r video_eval\requirements-ref.txt
   curl -L -o models/face/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   venv_ref\Scripts\python video_eval\reference_eyes.py --video Videos\IMG_0566.MOV --out video_eval\recordings\P01
   ```

2. **Scenario windows** (`P01_segments.csv`: `start_s,end_s,type`) written by
   hand from the spoken cues and the video. Types: `baseline` (natural gaze),
   `closure_phase`, `head_roll`, `head_down`, `head_up`, `slow_blinks`,
   `half_closed`, `yawn`, `rub_eyes` (`head_left`, `head_right` for P06).
3. **Labels:** `python video_eval/make_labels.py --session P01 --video Videos/IMG_0566.MOV`.
   A frame is closed when the mean MediaPipe eye-blink score exceeds 0.5. Closures
   of at least 0.3 s inside `closure_phase` become `closure` events; shorter
   ones in natural gaze become `blink`; longer uncued ones in natural gaze
   become `long_closure`. Check every `closure` in `P01_events.png` (middle
   frame of each event) before scoring.

Onset is the first frame the reference marks closed (blink score > 0.5), which
is later than the first lid movement; latencies can therefore be slightly
shorter than the counter's theoretical minimum.

## 4. Processing

**Event-level metrics at camera rate.** Every frame is processed offline, so this is independent of GPU speed:

```
python video_eval/run_pipeline.py --video video_eval/recordings/P01_normal_<timestamp>.mp4
```

**The same, as a slower live system would see it.** Use the frame rate measured in step 5; frames the system would have been too busy for are skipped:

```
python video_eval/run_pipeline.py --video ... --simulate-fps <measured fps>
```

For phone recordings add `--resize 1280x720` (the prototype's webcam
resolution; the aspect ratio must match).

**Score all sessions:**

```
python video_eval/score_events.py --dir video_eval/recordings
```

False alarms are counted only in `baseline` windows (natural gaze); for every
other scenario window the scorer reports whether an alert was raised
(`alert_rate_by_segment`). Latency is computed only for closures that begin
with no alert active.

**Alert-module output (emulated).** Replays the PC states through the
firmware's state logic (serial transfer neglected):

```
python video_eval/emulate_firmware.py --dst video_eval/recordings_module          # original firmware
python video_eval/emulate_firmware.py --fixed --dst video_eval/recordings_module_fixed  # released firmware
python video_eval/score_events.py --dir video_eval/recordings_module_fixed
```

## 5. Throughput and end-to-end latency (live)

Run it with the laptop plugged in and cool, and note the GPU. Screenshots from an overheating laptop are not representative.

```
python video_eval/run_pipeline.py --camera 0 --seconds 120 --serial COM5
```

- The output JSON gives the processed frame rate and per-stage latency (face detection, cropping, ViT inference with GPU synchronization, total) as mean, median and 95th percentile.
- `serial_acks` gives the time at which the Arduino confirmed each state change. The difference from `sent_state_at` in the CSV is the serial and firmware delay, including the firmware's minimum dwell times (Warning at least 1 s before escalating to Critical and 2 s before returning to Normal; Critical at least 1 s).

## 6. What to report in the paper

- **Participants and data:** number of participants, sessions, closure events and hours of recording.
- **Detection:** Warning and Critical detection rate by closure duration (<0.75, 0.75–1.5, 1.5–2.5, 2.5–4, ≥4 s).
- **Latency:** time from closure onset to Warning and to Critical (median and 95th percentile), at camera rate and at the measured live frame rate.
- **False alarms:** Warning onsets not linked to a closure, per hour; state how many were caused by annotated spontaneous blinks.
- **Face detection:** availability overall and per head pose.
- **Throughput:** processed frame rate and end-to-end latency (median and 95th percentile), with the hardware used.
- **Limitation:** these are posed closures by alert participants in a lab, not naturally occurring drowsiness in a vehicle.
