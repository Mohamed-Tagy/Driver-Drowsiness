"""
Event-level scoring of the prototype on annotated recordings.

For every session it pairs <session>_pipeline.csv (from run_pipeline.py)
with the annotation file <session>_labels.csv (checked by a person), or,
if that does not exist yet, the draft <session>_cues.csv (reported as
DRAFT). Annotation rows: start_s, end_s, type. Types starting with
"closure" are eye-closure events; "head_*" rows mark head-pose segments;
"baseline" and "open" are normal driving.

Metrics
  * detection rate of closure events, by annotated duration, for the
    Warning and Critical states (alert reached between closure onset and
    offset + grace period);
  * alert latency from closure onset (mean, median, 95th percentile);
  * false alarms: Warning onsets not linked to any closure event, per hour;
  * face-detection availability per segment type (e.g. head poses);
  * frame-level closed-eye recall inside closure events;
  * processing latency and processed frame rate (from the pipeline JSON).

Usage (standard library only):
    python video_eval/score_events.py --dir video_eval/recordings
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="video_eval/recordings")
    p.add_argument("--grace", type=float, default=1.0,
                   help="Seconds after closure offset in which an alert still counts")
    return p.parse_args()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(values, q):
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summary(values):
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.mean(values), 3),
            "median": round(pct(values, 50), 3), "p95": round(pct(values, 95), 3)}


def score_session(frames, labels, grace):
    t = [float(r["t"]) for r in frames]
    state = [int(r["state"]) for r in frames]
    closed = [int(r["frame_closed"]) for r in frames]
    face = [int(r["face"]) for r in frames]
    duration_s = t[-1] - t[0] if len(t) > 1 else 0.0

    events = [dict(start=float(r["start_s"]), end=float(r["end_s"]), type=r["type"])
              for r in labels if r["type"].startswith("closure")]
    # Alert onsets = frames where the state rises.
    onsets = {1: [], 2: []}
    prev = 0
    for ti, s in zip(t, state):
        for level in (1, 2):
            if s >= level > prev:
                onsets[level].append(ti)
        prev = s

    per_event, linked_warnings = [], set()
    for ev in events:
        win_end = ev["end"] + grace
        rec = {"type": ev["type"], "duration": round(ev["end"] - ev["start"], 3)}
        for level, name in ((1, "warning"), (2, "critical")):
            hit = next((ti for ti, s in zip(t, state)
                        if ev["start"] <= ti <= win_end and s >= level), None)
            rec[f"{name}_detected"] = hit is not None
            rec[f"{name}_latency_s"] = round(hit - ev["start"], 3) if hit is not None else None
        for i, o in enumerate(onsets[1]):
            if ev["start"] <= o <= win_end:
                linked_warnings.add(i)
        in_ev = [c for ti, c in zip(t, closed) if ev["start"] <= ti <= ev["end"]]
        rec["frames"] = len(in_ev)
        rec["frames_closed"] = sum(in_ev)
        per_event.append(rec)

    false_alarms = [o for i, o in enumerate(onsets[1]) if i not in linked_warnings]
    blinks = [(float(r["start_s"]), float(r["end_s"])) for r in labels
              if r["type"] == "blink"]
    fa_blink = [o for o in false_alarms
                if any(s <= o <= e + grace for s, e in blinks)]
    closure_time = sum(e["end"] - e["start"] + grace for e in events)

    segments = defaultdict(lambda: [0, 0])
    for r in labels:
        s0, s1 = float(r["start_s"]), float(r["end_s"])
        for ti, f in zip(t, face):
            if s0 <= ti <= s1:
                segments[r["type"]][0] += f
                segments[r["type"]][1] += 1
    return {"duration_s": round(duration_s, 1), "events": per_event,
            "false_alarm_times_s": [round(x, 2) for x in false_alarms],
            "false_alarms_during_annotated_blinks": len(fa_blink),
            "non_event_hours": max(0.0, duration_s - closure_time) / 3600,
            "face_by_segment": {k: v for k, v in segments.items()},
            "face_overall": [sum(face), len(face)]}


def duration_bin(d):
    for edge in (0.75, 1.5, 2.5, 4.0):
        if d < edge:
            return f"<{edge:g}s"
    return ">=4s"


def main():
    args = parse_args()
    root = Path(args.dir)
    sessions = {}
    for pcsv in sorted(root.glob("*_pipeline.csv")):
        name = pcsv.name[: -len("_pipeline.csv")]
        labels = root / f"{name}_labels.csv"
        draft = not labels.exists()
        if draft:
            labels = root / f"{name}_cues.csv"
        if not labels.exists():
            print(f"skip {name}: no labels or cues file")
            continue
        res = score_session(read_csv(pcsv), read_csv(labels), args.grace)
        res["draft_labels"] = draft
        pj = pcsv.with_suffix(".json")
        if pj.exists():
            res["pipeline"] = json.loads(pj.read_text())
        sessions[name] = res
    if not sessions:
        raise SystemExit(f"No *_pipeline.csv files in {root}")

    events = [e for s in sessions.values() for e in s["events"]]
    by_bin = defaultdict(list)
    for e in events:
        by_bin[duration_bin(e["duration"])].append(e)
    hours = sum(s["non_event_hours"] for s in sessions.values())
    n_fa = sum(len(s["false_alarm_times_s"]) for s in sessions.values())
    face_seg = defaultdict(lambda: [0, 0])
    for s in sessions.values():
        for k, (a, b) in s["face_by_segment"].items():
            face_seg[k][0] += a
            face_seg[k][1] += b
    total = {
        "sessions": len(sessions),
        "draft_label_sessions": sum(s["draft_labels"] for s in sessions.values()),
        "closure_events": len(events),
        "detection_by_duration": {
            b: {"n": len(v),
                "warning_rate": round(sum(e["warning_detected"] for e in v) / len(v), 3),
                "critical_rate": round(sum(e["critical_detected"] for e in v) / len(v), 3)}
            for b, v in sorted(by_bin.items())},
        "warning_latency_s": summary([e["warning_latency_s"] for e in events
                                      if e["warning_latency_s"] is not None]),
        "critical_latency_s": summary([e["critical_latency_s"] for e in events
                                       if e["critical_latency_s"] is not None]),
        "false_alarms": n_fa,
        "false_alarms_during_annotated_blinks": sum(
            s["false_alarms_during_annotated_blinks"] for s in sessions.values()),
        "non_event_hours": round(hours, 3),
        "false_alarms_per_hour": round(n_fa / hours, 2) if hours else None,
        "closed_frame_recall_in_events": round(
            sum(e["frames_closed"] for e in events) /
            max(1, sum(e["frames"] for e in events)), 4),
        "face_available_by_segment": {k: round(a / b, 4) for k, (a, b)
                                      in sorted(face_seg.items()) if b},
        "processed_fps": [s["pipeline"].get("processed_fps") for s in sessions.values()
                          if "pipeline" in s],
        "total_latency_ms": [s["pipeline"]["latency_ms"]["ms_total"]
                             for s in sessions.values() if "pipeline" in s],
    }
    (root / "event_scores.json").write_text(json.dumps(
        {"overall": total, "sessions": sessions}, indent=2))

    print(f"Sessions: {total['sessions']} (draft labels: {total['draft_label_sessions']})"
          f" | closure events: {total['closure_events']}")
    print("Detection by closure duration (Warning / Critical):")
    for b, v in total["detection_by_duration"].items():
        print(f"  {b:>6}: n={v['n']:3d}  {v['warning_rate']:.0%} / {v['critical_rate']:.0%}")
    print(f"Latency to Warning  (s): {total['warning_latency_s']}")
    print(f"Latency to Critical (s): {total['critical_latency_s']}")
    print(f"False alarms: {n_fa} in {hours:.2f} h without closures -> "
          f"{total['false_alarms_per_hour']} per hour")
    print(f"Closed-frame recall inside closures: {total['closed_frame_recall_in_events']}")
    print(f"Face detected by segment: {total['face_available_by_segment']}")
    if total["draft_label_sessions"]:
        print("WARNING: some sessions were scored against DRAFT cue labels; "
              "check them against the video before reporting.")
    print(f"Wrote {root / 'event_scores.json'}")


if __name__ == "__main__":
    main()
