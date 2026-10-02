"""
Unit tests for the analysis code behind the paper's statistics and
video-evaluation numbers.

    python -m pytest tests -q
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# project layout (subject_disjoint/, video_eval/) or released repository
# layout (code/subject_disjoint/, code/video_eval/)
CODE = ROOT / "code" if (ROOT / "code" / "video_eval").exists() else ROOT
sys.path.insert(0, str(CODE / "subject_disjoint"))
sys.path.insert(0, str(CODE / "video_eval"))

import emulate_firmware  # noqa: E402
import kfold_summary  # noqa: E402
import paired_stats  # noqa: E402
import score_events  # noqa: E402
import csv  # noqa: E402

import pytest  # noqa: E402

FPS = 30.0


# --- statistics -------------------------------------------------------------

def test_wilson_matches_paper_interval():
    lo, hi = paired_stats.wilson(16851, 16981)
    assert round(100 * lo, 2) == 99.09
    assert round(100 * hi, 2) == 99.35


def test_mcnemar_exact_two_sided():
    a = {f"x{i}": ("s", 1) for i in range(3)}
    b = {f"x{i}": ("s", 0) for i in range(3)}
    n01, n10, p, n = paired_stats.mcnemar(a, b)
    assert (n01, n10, n) == (3, 0, 3)
    assert abs(p - 0.25) < 1e-12           # 2 * (1/2)^3
    same = {f"x{i}": ("s", i % 2) for i in range(10)}
    assert paired_stats.mcnemar(same, same)[2] == 1.0


def test_cluster_bootstrap_brackets_accuracy():
    pred = {f"{s}_{i}": (s, int(i < 95)) for s in ("a", "b", "c", "d") for i in range(100)}
    lo, hi = paired_stats.cluster_boot(pred)
    assert lo <= 0.95 <= hi


# --- firmware emulation -----------------------------------------------------

def frames(states):
    return [{"t": f"{i / FPS:.4f}", "state": str(s)} for i, s in enumerate(states)]


def module_states(states, fixed):
    return [r["state"] for r in emulate_firmware.emulate(frames(states), fixed)]


def first(seq, value):
    return next(i for i, s in enumerate(seq) if s == value)


def test_warning_escalates_to_critical_after_one_second():
    pc = [0] * 30 + [1] * 5 + [2] * 120 + [0] * 30
    mod = module_states(pc, fixed=True)
    assert first(mod, 1) == 30                      # Warning immediately
    assert first(mod, 2) - first(mod, 1) == 30      # Critical 1 s later


def test_warning_held_two_seconds_before_normal():
    pc = [0] * 10 + [1] * 5 + [0] * 120
    mod = module_states(pc, fixed=True)
    w = first(mod, 1)
    back = next(i for i in range(w, len(mod)) if mod[i] == 0)
    assert back - w == 60                           # 2 s at 30 fps


def test_queued_critical_bug_and_fix():
    # PC: Warning at frame 10, Critical at 15, eyes reopen so the PC is back
    # in Warning at frame 36 and in Normal at 42. The module's 1-s Warning
    # dwell ends at frame 40, while the PC is in Warning again.
    pc = [0] * 10 + [1] * 5 + [2] * 21 + [1] * 6 + [0] * 120
    orig = module_states(pc, fixed=False)
    fix = module_states(pc, fixed=True)
    assert first(orig, 2) == 40  # original: queued Critical fires after reopening
    assert 2 not in fix          # corrected: PC's Warning cancels the queued change

    # If the PC is already back in Normal when the dwell ends, the queued
    # Critical is replaced by Normal in both versions.
    pc = [0] * 10 + [1] * 5 + [2] * 5 + [1] * 5 + [0] * 120
    assert 2 not in module_states(pc, fixed=False)


# --- event scoring ----------------------------------------------------------

def test_latency_only_from_normal_and_value():
    n = 300
    state = [0] * n
    closed = [0] * n
    for i in range(60, 120):            # closure 2.0-4.0 s
        closed[i] = 1
    for i in range(70, 125):            # Warning 10 frames after onset
        state[i] = 1
    for i in range(75, 125):            # Critical 15 frames after onset
        state[i] = 2
    rows = [{"t": f"{i / FPS:.4f}", "state": str(state[i]),
             "frame_closed": str(closed[i]), "face": "1"} for i in range(n)]
    labels = [{"start_s": "2.0", "end_s": "4.0", "type": "closure"},
              {"start_s": "0.0", "end_s": "1.9", "type": "baseline"}]
    res = score_events.score_session(rows, labels, grace=1.0)
    ev = res["events"][0]
    assert ev["warning_detected"] and ev["critical_detected"]
    assert abs(ev["warning_latency_s"] - 10 / FPS) < 1e-3
    assert abs(ev["critical_latency_s"] - 15 / FPS) < 1e-3
    assert res["false_alarms_baseline"] == 0

    # Same closure starting while an alert is already on: detected, no latency.
    for i in range(55, 70):
        rows[i]["state"] = "1"
    res = score_events.score_session(rows, labels, grace=1.0)
    ev = res["events"][0]
    assert ev["warning_detected"] and ev["warning_latency_s"] is None


# --- k-fold cross-validation ------------------------------------------------

def test_sign_test_and_median():
    assert kfold_summary.sign_test([1] * 10)[2] == pytest.approx(2 / 1024)
    assert kfold_summary.sign_test([1, -1])[2] == 1.0
    assert kfold_summary.sign_test([0, 0, 0]) == (0, 0, 1.0)
    assert kfold_summary.median([3, 1, 2]) == 2
    assert kfold_summary.median([4, 1, 2, 3]) == 2.5


def _write_preds(path, subject, n=20):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "subject", "label", "pred", "prob_closed"])
        for i in range(n):
            lab = "closed" if i % 2 else "open"
            w.writerow([f"test/x/{subject}_{i:05d}.png", subject, lab, lab, "0.5"])


def test_partial_run_is_not_pooled(tmp_path):
    for k in range(2):           # only 2 of 5 folds finished
        _write_preds(tmp_path / f"kfold5_fold{k}_vit-base_base_strong_seed42"
                     / "predictions_test.csv", f"s{k:04d}")
    out = kfold_summary.summarize(runs=tmp_path, manifests=tmp_path / "none",
                                  control=tmp_path / "missing.csv")
    assert out["folds_complete"] == 2 and not out["complete"]
    assert "pooled" not in out


def test_kfold_manifest_invariants():
    mdir = ROOT / "subject_disjoint" / "manifests"
    if not mdir.exists():
        mdir = ROOT / "manifests" / "kfold5"
    files = sorted(mdir.glob("kfold5_fold*.csv"))
    if len(files) != 5:
        pytest.skip("k-fold manifests not present")
    tested, paths0 = [], None
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        paths = sorted(r["path"] for r in rows)
        assert len(paths) == len(set(paths))
        paths0 = paths0 or paths
        assert paths == paths0                     # same image set in every fold
        part = {}
        for r in rows:
            assert part.setdefault(r["subject"], r["split"]) == r["split"]
        tested += [s for s, sp in part.items() if sp == "test"]
    assert len(tested) == len(set(tested)) == 37   # every subject tested once
