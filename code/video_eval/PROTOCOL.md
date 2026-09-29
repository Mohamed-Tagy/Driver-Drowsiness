# System-level evaluation protocol

Purpose: measure what the instructor asked for (comments #1, #4, #16, #23, #28):
- event-level detection of sustained eye closures;
- alert latency from the onset of eye closure;
- false alarms per hour;
- face-detection failures by head pose;
- end-to-end frame rate and latency.

None of this needs model training. It uses the selected checkpoint
`models/vit-base-mrl-augmented-2/final_model` and the same logic as `demo.py`.

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

The script runs for about 6 minutes. It shows cues, records the raw video (without the cue overlay), and saves capture timestamps and cue intervals. The cue order is randomized per `--seed`; use a different seed for each participant.

**Script (per session):**
1. 60 s of normal driving gaze with natural blinking.
2. 15 instructed eye closures (0.5, 1, 2, 3 and 5 s, three of each, random order), separated by 6–10 s of eyes open.
3. Head poses held for 5 s each: about 30° left and right, about 20° down and up.
4. 60 s of normal driving gaze.

The participant should keep looking toward the road (the camera area) except during head-pose cues.

## 3. Annotation (required before reporting)

`<session>_cues.csv` records what was *instructed*, not what happened: people close their eyes a few hundred milliseconds after the cue and blink spontaneously. For each session, create `<session>_labels.csv` with the same columns (`start_s,end_s,type`):

1. Copy `<session>_cues.csv` to `<session>_labels.csv`.
2. Open the video in a player that shows the time, e.g. VLC with the time display in milliseconds, or step through frames and use `<session>_frames.csv` to convert frame numbers to seconds.
3. For every `closure_*` row, set `start_s` to the first frame with the eyelids fully covering the pupil and `end_s` to the first frame they open again. Keep the `type`.
4. Add a row `start_s,end_s,blink` for every spontaneous blink or closure outside the cued closures. Blinks are reported separately and are *not* counted as misses.
5. Ideally, a second person annotates two sessions independently; report the mean absolute difference in onset times.

The scorer uses `_labels.csv` when it exists, and otherwise falls back to `_cues.csv`, flagging the results as DRAFT.

## 4. Processing

**Event-level metrics at camera rate.** Every frame is processed offline, so this is independent of GPU speed:

```
python video_eval/run_pipeline.py --video video_eval/recordings/P01_normal_<timestamp>.mp4
```

**The same, as a slower live system would see it.** Use the frame rate measured in step 5; frames the system would have been too busy for are skipped:

```
python video_eval/run_pipeline.py --video ... --simulate-fps <measured fps>
```

**Score all sessions:**

```
python video_eval/score_events.py --dir video_eval/recordings
```

## 5. Throughput and end-to-end latency (live)

Run it with the laptop plugged in and cool, and note the GPU. Screenshots from an overheating laptop are not representative.

```
python video_eval/run_pipeline.py --camera 0 --seconds 120 --serial COM5
```

- The output JSON gives the processed frame rate and per-stage latency (face detection, cropping, ViT inference with GPU synchronization, total) as mean, median and 95th percentile.
- `serial_acks` gives the time at which the Arduino confirmed each state change. The difference from `sent_state_at` in the CSV is the serial and firmware delay, including the firmware's minimum dwell times (2 s in Warning, 1 s in Critical).

## 6. What to report in the paper

- **Participants and data:** number of participants, sessions, closure events and hours of recording.
- **Detection:** Warning and Critical detection rate by closure duration (<0.75, 0.75–1.5, 1.5–2.5, 2.5–4, ≥4 s).
- **Latency:** time from closure onset to Warning and to Critical (median and 95th percentile), at camera rate and at the measured live frame rate.
- **False alarms:** Warning onsets not linked to a closure, per hour; state how many were caused by annotated spontaneous blinks.
- **Face detection:** availability overall and per head pose.
- **Throughput:** processed frame rate and end-to-end latency (median and 95th percentile), with the hardware used.
- **Limitation:** these are posed closures by alert participants in a lab, not naturally occurring drowsiness in a vehicle.
