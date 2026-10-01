"""
Emulate the Arduino module's displayed state from logged PC states.

Replays the PC alert-state sequence of each *_pipeline.csv through the
logic of arduino/drowsiness_receiver/drowsiness_receiver.ino, exactly as
coded:
  * the PC sends a state only when it changes (demo.py);
  * a received state different from the module's current state becomes
    the pending state (a state equal to the current one is ignored and
    does NOT clear an older pending state);
  * Normal -> any: immediately; Warning -> Critical: after >= 1 s in
    Warning; Warning -> Normal: after >= 2 s; Critical -> any: after >= 1 s.
Serial transfer (2-3 bytes at 9600 baud, ~3 ms) and loop delays are
neglected. The output keeps every column of the input but replaces
"state" with the module state, so score_events.py can score it.

    python video_eval/emulate_firmware.py --src video_eval/recordings \
        --dst video_eval/recordings_module
"""

import argparse
import csv
import shutil
from pathlib import Path

MIN_WARNING, MIN_CRITICAL = 2.0, 1.0


def emulate(rows, fixed=False):
    cur, pending, entered, sent_last, t_recv = 0, None, 0.0, None, 0.0
    out = []
    for r in rows:
        t, pc = float(r["t"]), int(r["state"])
        if pc != sent_last:                      # PC sends on change only
            sent_last = pc
            if pc != cur:
                pending, t_recv = pc, t
            elif fixed:                          # corrected firmware: a state
                pending = None                   # equal to the current one
                                                 # cancels an older pending one
        if pending is not None:
            if cur == 0:
                need = 0.0
            elif cur == 1:
                need = MIN_CRITICAL if pending == 2 else MIN_WARNING
            else:
                need = MIN_CRITICAL
            if t - entered >= need:
                # The firmware polls continuously: the switch happened when
                # both the message had arrived and the dwell had elapsed.
                cur, pending, entered = pending, None, max(t_recv, entered + need)
        out.append(dict(r, state=cur, pc_state=pc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="video_eval/recordings")
    ap.add_argument("--dst", default="video_eval/recordings_module")
    ap.add_argument("--fixed", action="store_true",
                    help="emulate the corrected firmware (pending state cleared "
                         "when the PC returns to the current state)")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("*_pipeline.csv")):
        name = p.name[: -len("_pipeline.csv")]
        labels = src / f"{name}_labels.csv"
        if not labels.exists():
            continue
        rows = emulate(list(csv.DictReader(open(p))), a.fixed)
        with open(dst / p.name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        shutil.copy(labels, dst / labels.name)
        print(f"{name}: module states written")


if __name__ == "__main__":
    main()
