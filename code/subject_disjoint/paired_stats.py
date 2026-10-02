"""
Paired and subject-level statistics on the original test partition.

Reads per-image predictions written by evaluate_checkpoints.py (and the
control run of train_subject_disjoint.py) and reports, for each model:
  * accuracy with a 95% Wilson interval (images treated as independent);
  * a 95% subject-cluster bootstrap interval (resampling the 37 subjects,
    which accounts for correlated images of the same person);
and for each pair of models an exact two-sided McNemar test on the
discordant images.

    python subject_disjoint/paired_stats.py
Output: subject_disjoint/checkpoint_eval/paired_stats.json
"""

import csv
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROL_RUN = Path("kaggle_split_vit-base_base_strong_seed42") / "predictions_test.csv"
# Project layout (subject_disjoint/checkpoint_eval, subject_disjoint/runs) or
# released-repository layout (checkpoint_eval/, subject_disjoint_runs/ at the
# top level, script in code/subject_disjoint/).
if (HERE / "checkpoint_eval").exists():
    EVAL, CONTROL = HERE / "checkpoint_eval", HERE / "runs" / CONTROL_RUN
else:
    TOP = HERE.parent.parent
    EVAL, CONTROL = TOP / "checkpoint_eval", TOP / "subject_disjoint_runs" / CONTROL_RUN
MODELS = {"exp1": "ViT-Tiny, MRL (Exp. 1)", "exp2": "ViT-Base, MRL (Exp. 2)",
          "exp3": "ViT-Base, MRL Aug. (Exp. 3, selected)",
          "exp3earlier": "ViT-Base, MRL Aug., earlier run (patience 3)",
          "control": "ViT-Base, MRL, new script (control)"}
B = 10000


def load(name):
    if name == "control":
        with open(CONTROL, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {Path(r["path"]).name: (r["subject"], int(r["label"] == r["pred"]))
                for r in rows}
    with open(EVAL / f"{name}_test_predictions.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["file"]: (r["subject"], int(r["label"] == r["pred"])) for r in rows}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cluster_boot(pred, seed=0):
    by = {}
    for subj, ok in pred.values():
        a = by.setdefault(subj, [0, 0])
        a[0] += ok
        a[1] += 1
    subs = list(by)
    rng = random.Random(seed)
    accs = []
    for _ in range(B):
        k = n = 0
        for s in (rng.choice(subs) for _ in subs):
            k += by[s][0]
            n += by[s][1]
        accs.append(k / n)
    accs.sort()
    return accs[int(0.025 * B)], accs[int(0.975 * B) - 1]


def mcnemar(a, b):
    keys = a.keys() & b.keys()
    n01 = sum(1 for k in keys if a[k][1] == 1 and b[k][1] == 0)
    n10 = sum(1 for k in keys if a[k][1] == 0 and b[k][1] == 1)
    n = n01 + n10
    tail = sum(math.comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n if n else 1.0
    return n01, n10, min(1.0, 2 * tail), len(keys)


def main():
    global EVAL, CONTROL
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=None, help="folder with *_test_predictions.csv")
    ap.add_argument("--control", default=None, help="control run predictions_test.csv")
    a = ap.parse_args()
    EVAL = Path(a.eval_dir) if a.eval_dir else EVAL
    CONTROL = Path(a.control) if a.control else CONTROL
    preds = {m: load(m) for m in MODELS
             if m == "control" or (EVAL / f"{m}_test_predictions.csv").exists()}
    out = {"models": {}, "mcnemar": []}
    for m, p in preds.items():
        k, n = sum(ok for _, ok in p.values()), len(p)
        lo, hi = wilson(k, n)
        blo, bhi = cluster_boot(p)
        out["models"][m] = {"label": MODELS[m], "n": n, "correct": k, "acc": k / n,
                            "wilson95": [lo, hi], "subject_bootstrap95": [blo, bhi]}
        print(f"{m:12s} {100 * k / n:.2f}%  Wilson {100 * lo:.2f}-{100 * hi:.2f}  "
              f"subject-bootstrap {100 * blo:.2f}-{100 * bhi:.2f}  (n={n})")
    names = list(preds)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n01, n10, p, n = mcnemar(preds[names[i]], preds[names[j]])
            out["mcnemar"].append({"a": names[i], "b": names[j], "a_only_correct": n01,
                                   "b_only_correct": n10, "p_exact": p, "n_paired": n})
            print(f"McNemar {names[i]} vs {names[j]}: {n01} vs {n10} discordant, "
                  f"p = {p:.3g} (n={n})")
    (EVAL / "paired_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
