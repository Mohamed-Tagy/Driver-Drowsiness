"""
Summarize subject-disjoint runs: mean +/- SD across seeds and paired
McNemar tests between configurations evaluated on the same test images.

Usage (standard library only):
    python aggregate_results.py --runs subject_disjoint/runs

Writes summary.md and summary.json next to the runs, and prints LaTeX
table rows for the paper.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="subject_disjoint/runs")
    return p.parse_args()


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_predictions(run_dir):
    with open(run_dir / "predictions_test.csv", newline="", encoding="utf-8") as f:
        return {r["path"]: r["label"] == r["pred"] for r in csv.DictReader(f)}


def mean_sd(values):
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.mean(values), statistics.stdev(values)


def main():
    args = parse_args()
    root = Path(args.runs)
    runs = []
    for m in sorted(root.glob("*/metrics.json")):
        data = json.loads(m.read_text())
        cfg = data["config"]
        if cfg.get("smoke"):
            continue
        runs.append({"dir": m.parent, "seed": cfg["seed"],
                     "manifest": Path(cfg["manifest"]).name,
                     "key": f'{"Kaggle image-level" if "kaggle" in Path(cfg["manifest"]).name else "subject-disjoint"}'
                            f' / {cfg["model"]} / {cfg["train_set"]} / {cfg["online_aug"]}',
                     "test": data["test"], "val": data["val"],
                     "best_epoch": data["best_epoch"],
                     "subjects_test": len(data["subjects"]["test"])})
    if not runs:
        raise SystemExit(f"No finished runs (metrics.json) under {root}")

    groups = defaultdict(list)
    for r in runs:
        groups[r["key"]].append(r)

    lines = ["# Subject-disjoint results", "",
             "| Model / training set / online aug. | Seeds | Test acc. (mean ± SD) "
             "| Min–max | Macro F1 | Closed→open rate | Val acc. |",
             "|---|---|---|---|---|---|---|"]
    latex, summary = [], {}
    for key, rs in sorted(groups.items()):
        acc = [r["test"]["accuracy"] for r in rs]
        f1 = [r["test"]["macro_f1"] for r in rs]
        fnr = [r["test"]["closed_as_open_rate"] for r in rs]
        val = [r["val"]["accuracy"] for r in rs]
        (ma, sa), (mf, sf), (mn, sn), (mv, sv) = map(mean_sd, (acc, f1, fnr, val))
        seeds = sorted(r["seed"] for r in rs)
        lines.append(f"| {key} | {seeds} | {ma:.2f} ± {sa:.2f} | "
                     f"{min(acc):.2f}–{max(acc):.2f} | {mf:.2f} ± {sf:.2f} | "
                     f"{mn:.2f} ± {sn:.2f} | {mv:.2f} ± {sv:.2f} |")
        latex.append(f"{key.replace(' / ', ' & ')} & {len(rs)} & "
                     f"{ma:.2f} $\\pm$ {sa:.2f} & {mf:.2f} $\\pm$ {sf:.2f} & "
                     f"{mn:.2f} $\\pm$ {sn:.2f} \\\\")
        summary[key] = {"seeds": seeds, "test_acc": acc, "test_acc_mean": ma,
                        "test_acc_sd": sa, "macro_f1": f1, "closed_as_open": fnr,
                        "val_acc": val,
                        "per_run": [{"seed": r["seed"], "best_epoch": r["best_epoch"],
                                     "test_subjects": r["subjects_test"],
                                     "per_subject_accuracy":
                                         r["test"]["per_subject_accuracy"]}
                                    for r in rs]}

    lines += ["", "## Paired McNemar tests (same split, same test images)", "",
              "| A | B | Seed | A right / B wrong | A wrong / B right | p (exact) |",
              "|---|---|---|---|---|---|"]
    mcnemar = []
    for ka, kb in combinations(sorted(groups), 2):
        by_seed_b = {(r["seed"], r["manifest"]): r for r in groups[kb]}
        for ra in groups[ka]:
            rb = by_seed_b.get((ra["seed"], ra["manifest"]))
            if not rb:
                continue
            pa, pb = load_predictions(ra["dir"]), load_predictions(rb["dir"])
            common = pa.keys() & pb.keys()
            b = sum(pa[k] and not pb[k] for k in common)
            c = sum(pb[k] and not pa[k] for k in common)
            p = mcnemar_exact(b, c)
            mcnemar.append({"A": ka, "B": kb, "seed": ra["seed"], "b": b, "c": c,
                            "p": p, "n": len(common)})
            lines.append(f"| {ka} | {kb} | {ra['seed']} | {b} | {c} | {p:.4g} |")
    if not mcnemar:
        lines.append("| (needs two configurations run on the same split) |||||| ")

    out_md = root / "summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(
        {"groups": summary, "mcnemar": mcnemar}, indent=2))
    print("\n".join(lines))
    print("\nLaTeX rows (model & train set & online aug. & seeds & acc. & "
          "macro F1 & closed->open):")
    print("\n".join(latex))
    print(f"\nWritten: {out_md}")


if __name__ == "__main__":
    main()
