"""
Summarize the grouped 5-fold subject-disjoint cross-validation.

Reads runs/kfold5_fold<k>_vit-base_base_strong_seed42/predictions_test.csv
(written by train_subject_disjoint.py), the fold manifests, and the
image-level control run, and reports:
  * per fold: subjects, test images, accuracy, closed->open, open->closed;
  * pooled over all folds (every subject tested exactly once): accuracy with
    a subject-cluster bootstrap 95% interval (it reflects which subjects are
    sampled, not training-run variance: one seed per fold), both error rates;
  * per subject, paired on images: accuracy when unseen (its test fold) vs
    seen (control run) on the SAME images (the subject's images in the
    control test set), with an exact two-sided sign test over subjects; the
    unseen accuracy on all of a subject's images is reported descriptively.

Pooled and per-subject results are written only when all K folds are
complete and each fold's test subjects match its manifest (use
--allow-partial to inspect unfinished runs; the output is then flagged).

    python subject_disjoint/kfold_summary.py
Output: runs/kfold5_summary.json and a printed table.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# project layout (subject_disjoint/runs, subject_disjoint/manifests) or the
# released repository layout (subject_disjoint_runs/, manifests/ at the top)
if (HERE / "runs").exists():
    RUNS, MANIFESTS = HERE / "runs", HERE / "manifests"
else:
    RUNS, MANIFESTS = HERE.parent.parent / "subject_disjoint_runs", HERE.parent.parent / "manifests" / "kfold5"
CONTROL_RUN = Path("kaggle_split_vit-base_base_strong_seed42") / "predictions_test.csv"
K = 5
B = 10000


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rates(rows):
    n = len(rows)
    acc = sum(r["label"] == r["pred"] for r in rows) / n
    closed = [r for r in rows if r["label"] == "closed"]
    opened = [r for r in rows if r["label"] == "open"]
    co = sum(r["pred"] == "open" for r in closed) / len(closed)
    oc = sum(r["pred"] == "closed" for r in opened) / len(opened)
    return acc, co, oc


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        raise ValueError("median of empty list")
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def sign_test(diffs):
    """Exact two-sided sign test; ties (zero differences) are dropped."""
    neg = sum(x < 0 for x in diffs)
    pos = sum(x > 0 for x in diffs)
    n = neg + pos
    if n == 0:
        return 0, 0, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(neg, pos) + 1)) / 2 ** n
    return neg, pos, min(1.0, 2 * tail)


def acc_by_subject(rows):
    d = defaultdict(lambda: [0, 0])
    for r in rows:
        d[r["subject"]][0] += r["label"] == r["pred"]
        d[r["subject"]][1] += 1
    return {s: (k, n) for s, (k, n) in d.items()}


def cluster_boot(by_subject, seed=0):
    subs = list(by_subject)
    rng = random.Random(seed)
    boot = []
    for _ in range(B):
        k = n = 0
        for s in (rng.choice(subs) for _ in subs):
            k += by_subject[s][0]
            n += by_subject[s][1]
        boot.append(k / n)
    boot.sort()
    return boot[int(0.025 * B)], boot[int(0.975 * B) - 1]


def summarize(runs=RUNS, manifests=MANIFESTS, control=None, allow_partial=False):
    control = control or runs / CONTROL_RUN
    folds, pooled, problems = [], [], []
    for k in range(K):
        run = runs / f"kfold5_fold{k}_vit-base_base_strong_seed42"
        pf = run / "predictions_test.csv"
        if not pf.exists():
            problems.append(f"fold {k}: no predictions_test.csv (not finished or failed)")
            continue
        rows = load(pf)
        subs = {r["subject"] for r in rows}
        mf = manifests / f"kfold5_fold{k}.csv"
        if mf.exists():
            expected = {r["subject"] for r in load(mf) if r["split"] == "test"}
            if subs != expected:
                problems.append(f"fold {k}: test subjects differ from {mf.name}")
        m = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else {}
        acc, co, oc = rates(rows)
        folds.append({"fold": k, "subjects": len(subs), "images": len(rows),
                      "acc": acc, "closed_to_open": co, "open_to_closed": oc,
                      "best_epoch": m.get("best_epoch"),
                      "training_minutes": m.get("training_minutes")})
        pooled += rows
    complete = len(folds) == K and not problems
    out = {"folds_complete": len(folds), "folds_expected": K, "complete": complete,
           "problems": problems, "folds": folds}
    if not folds:
        return out
    accs = [f["acc"] * 100 for f in folds]
    mean = sum(accs) / len(accs)
    sd = (sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5 if len(accs) > 1 else 0.0
    out.update({"fold_mean_acc": mean, "fold_sd_acc": sd})
    if not complete and not allow_partial:
        return out

    acc, co, oc = rates(pooled)
    by_s = acc_by_subject(pooled)
    lo, hi = cluster_boot(by_s)

    ctrl_rows = load(control)
    ctrl_by_path = {Path(r["path"]).name: r for r in ctrl_rows}
    ctrl_by_s = acc_by_subject(ctrl_rows)
    unseen_rows_by_s = defaultdict(list)
    for r in pooled:
        unseen_rows_by_s[r["subject"]].append(r)
    per, diffs = [], []
    for s in sorted(by_s):
        if s not in ctrl_by_s:
            raise SystemExit(f"subject {s} missing from the control run {control}")
        # paired on images: unseen predictions on this subject's control-test images
        same = [r for r in unseen_rows_by_s[s] if Path(r["path"]).name in ctrl_by_path]
        k_seen, n_seen = ctrl_by_s[s]
        k_unseen_same = sum(r["label"] == r["pred"] for r in same)
        if len(same) != n_seen:
            raise SystemExit(f"subject {s}: {len(same)} paired images vs {n_seen} in control")
        unseen_same = k_unseen_same / n_seen
        seen = k_seen / n_seen
        diffs.append(unseen_same - seen)
        per.append({"subject": s, "unseen_all_images": by_s[s][0] / by_s[s][1],
                    "n_all": by_s[s][1], "unseen_paired": unseen_same, "seen_paired": seen,
                    "n_paired": n_seen})
    neg, pos, p = sign_test(diffs)
    unseen_all = [x["unseen_all_images"] for x in per]
    out.update({
        "pooled": {"subjects": len(by_s), "images": len(pooled), "acc": acc,
                   "subject_bootstrap95": [lo, hi], "closed_to_open": co,
                   "open_to_closed": oc},
        "per_subject": per,
        "per_subject_summary": {
            "median_unseen_all": median(unseen_all), "min_unseen_all": min(unseen_all),
            "below_97_unseen_all": sum(x < 0.97 for x in unseen_all),
            "paired_lower_when_unseen": neg, "paired_higher_when_unseen": pos,
            "paired_ties": len(diffs) - neg - pos,
            "paired_median_change_points": 100 * median(diffs),
            "paired_min_change_points": 100 * min(diffs),
            "paired_mean_change_points": 100 * sum(diffs) / len(diffs),
            "sign_test_p": p}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true",
                    help="also pool unfinished runs (output flagged complete=false)")
    a = ap.parse_args()
    out = summarize(allow_partial=a.allow_partial)
    for f in out["folds"]:
        print(f"fold {f['fold']}: {f['subjects']} subj, {f['images']} img, "
              f"acc {100*f['acc']:.2f}%, C>O {100*f['closed_to_open']:.2f}%, "
              f"O>C {100*f['open_to_closed']:.2f}%, best epoch {f['best_epoch']}")
    for prob in out["problems"]:
        print("PROBLEM:", prob)
    print(f"folds complete: {out['folds_complete']}/{K}")
    if "pooled" not in out:
        print("Pooled results not computed (incomplete run; use --allow-partial to inspect).")
        sys.exit(1 if out["problems"] else 0)
    po, ps = out["pooled"], out["per_subject_summary"]
    print(f"fold accuracy {out['fold_mean_acc']:.2f} +/- {out['fold_sd_acc']:.2f}")
    print(f"pooled ({po['subjects']} subjects, {po['images']} images): acc {100*po['acc']:.2f}% "
          f"(subject bootstrap {100*po['subject_bootstrap95'][0]:.2f}-"
          f"{100*po['subject_bootstrap95'][1]:.2f}), C>O {100*po['closed_to_open']:.2f}%, "
          f"O>C {100*po['open_to_closed']:.2f}%")
    print(f"per subject (all images, unseen): median {100*ps['median_unseen_all']:.1f}%, "
          f"min {100*ps['min_unseen_all']:.1f}%, {ps['below_97_unseen_all']} below 97%")
    print(f"paired on control-test images: lower when unseen in {ps['paired_lower_when_unseen']}, "
          f"higher in {ps['paired_higher_when_unseen']}, ties {ps['paired_ties']}; median change "
          f"{ps['paired_median_change_points']:.2f} pts, mean {ps['paired_mean_change_points']:.2f}, "
          f"min {ps['paired_min_change_points']:.2f}; sign test p = {ps['sign_test_p']:.3g}")
    if not out["complete"]:
        print("WARNING: partial run -- do not report these pooled numbers.")
    name = "kfold5_summary.json" if out["complete"] else "kfold5_summary_PARTIAL.json"
    (RUNS / name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", RUNS / name)


if __name__ == "__main__":
    main()
