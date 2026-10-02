"""
Grouped (subject-level) K-fold cross-validation manifests for MRL.

The 37 subjects are divided into K groups with similar image counts and
closed-eye ratios (best of --trials random assignments). For fold k the test
partition is group k, the validation partition is group (k+1) mod K, and the
remaining K-2 groups form the training partition. Every subject is therefore
tested exactly once, and with K=5 the image proportions stay close to the
60/20/20 used elsewhere.

    python subject_disjoint/make_kfold_split.py \
        --manifest subject_disjoint/manifests/kaggle_split.csv --k 5 --seed 2026

Outputs (same CSV format as make_subject_split.py, usable with
train_subject_disjoint.py --manifest):
    manifests/kfold5_fold<k>.csv   path, subject, label, split
    manifests/kfold5_groups.json   subjects and counts per group and fold
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="subject_disjoint/manifests/kaggle_split.csv",
                    help="any manifest listing every image once (path, subject, label)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=50000)
    ap.add_argument("--out-dir", default="subject_disjoint/manifests")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    n_img = Counter(r["subject"] for r in rows)
    n_closed = Counter(r["subject"] for r in rows if r["label"] == "closed")
    subjects = sorted(n_img)
    total, total_closed = len(rows), sum(n_closed.values())
    target, ratio = total / a.k, total_closed / total

    rng = random.Random(a.seed)
    best, best_score = None, float("inf")
    for _ in range(a.trials):
        order = subjects[:]
        rng.shuffle(order)
        groups = [order[i::a.k] for i in range(a.k)]
        # greedy rebalance is unnecessary for a score-based search; score =
        # relative size deviation + closed-ratio deviation of the worst group
        score = 0.0
        for g in groups:
            n = sum(n_img[s] for s in g)
            c = sum(n_closed[s] for s in g)
            score = max(score, abs(n - target) / target + abs(c / n - ratio))
        if score < best_score:
            best, best_score = groups, score

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = {"k": a.k, "seed": a.seed, "score": round(best_score, 5), "groups": [], "folds": []}
    for g in best:
        n = sum(n_img[s] for s in g)
        info["groups"].append({"subjects": sorted(g), "images": n,
                               "closed_ratio": round(sum(n_closed[s] for s in g) / n, 4)})
    for k in range(a.k):
        test, val = set(best[k]), set(best[(k + 1) % a.k])
        split_of = {s: ("test" if s in test else "val" if s in val else "train")
                    for s in subjects}
        path = out / f"kfold{a.k}_fold{k}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "subject", "label", "split"])
            for r in rows:
                w.writerow([r["path"], r["subject"], r["label"], split_of[r["subject"]]])
        cnt = Counter(split_of[r["subject"]] for r in rows)
        subj = defaultdict(list)
        for s, sp in split_of.items():
            subj[sp].append(s)
        info["folds"].append({"fold": k, "manifest": path.name,
                              "subjects": {sp: len(v) for sp, v in subj.items()},
                              "images": dict(cnt)})
        print(f"fold {k}: subjects train/val/test = {len(subj['train'])}/"
              f"{len(subj['val'])}/{len(subj['test'])}, images = "
              f"{cnt['train']}/{cnt['val']}/{cnt['test']}")
    (out / f"kfold{a.k}_groups.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("worst-group score", round(best_score, 4))


if __name__ == "__main__":
    main()
