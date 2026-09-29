"""
Subject-disjoint train/val/test split of the MRL Eye Dataset.

MRL file names encode the subject, e.g. s0001_00014_0_0_0_0_0_01.png:
    s0001  subject ID
    00014  image ID
    0      gender       (0 man, 1 woman)
    0      glasses      (0 no, 1 yes)
    0      eye state    (0 closed, 1 open)
    0      reflections  (0 none, 1 small, 2 big)
    0      lighting     (0 bad, 1 good)
    01     sensor ID    (01 RealSense, 02 IDS, 03 Aptina)

For each seed, the 37 subjects are divided into train/val/test groups so
that no person appears in more than one partition. Among many random
subject assignments, the one closest to the target image fractions
(default 60/20/20) and to the overall closed-eye ratio is kept.

Usage (no GPU or PyTorch needed):
    python make_subject_split.py --data-root data/MRL --seeds 42 43 44
    python make_subject_split.py --data-root data/MRL --audit-only

Outputs (in --out-dir, default subject_disjoint/manifests):
    split_seed<S>.csv       path (relative to --data-root), subject, label,
                            split  (one row per image)
    split_seed<S>.json      subjects and image/class counts per partition
    kaggle_split_audit.json subject overlap of the existing folder split
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

NAME_RE = re.compile(
    r"(s\d{4})_(\d+)_([01])_([01])_([01])_([012])_([01])_(\d{2})\.\w+$",
    re.IGNORECASE,
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test"}
LABELS = ("open", "closed")          # index 0 = open, 1 = closed
FOLDER_LABEL = {"awake": "open", "open": "open", "open_eyes": "open",
                "sleepy": "closed", "closed": "closed", "closed_eyes": "closed"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="Folder containing the MRL images (any layout, "
                        "e.g. the Kaggle train/val/test/awake|sleepy folders)")
    p.add_argument("--out-dir", default="subject_disjoint/manifests")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--fractions", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--trials", type=int, default=20000,
                   help="Random subject assignments evaluated per seed")
    p.add_argument("--audit-only", action="store_true",
                   help="Only report subject overlap of the existing split")
    p.add_argument("--write-kaggle-manifest", action="store_true",
                   help="Also write kaggle_split.csv listing the existing "
                        "train/val/test folders (for the reproducibility archive)")
    return p.parse_args()


def scan(data_root: Path):
    """Return one record per unique image file name."""
    records, skipped, dupes = {}, [], 0
    label_mismatch = augmented = 0
    for path in sorted(data_root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        m = NAME_RE.search(path.name)
        if not m:
            skipped.append(str(path))
            continue
        prefix = path.name[:m.start()].lower()
        if prefix and prefix != "orig_":
            augmented += 1               # flip_, rot_pos_, ... offline copies
            continue
        subject = m.group(1).lower()
        name_label = "closed" if m.group(5) == "0" else "open"
        parts = [p.lower() for p in path.relative_to(data_root).parts[:-1]]
        folder_label = next((FOLDER_LABEL[p] for p in parts if p in FOLDER_LABEL), None)
        folder_split = next((SPLIT_ALIASES[p] for p in parts if p in SPLIT_ALIASES), None)
        if folder_label and folder_label != name_label:
            label_mismatch += 1
        key = f"{subject}_{m.group(2)}_{m.group(8)}"
        if key in records:
            dupes += 1
            continue
        records[key] = {
            "path": path.relative_to(data_root).as_posix(),
            "subject": subject,
            "label": folder_label or name_label,
            "folder_split": folder_split,
        }
    if augmented:
        print(f"Ignored {augmented:,} offline-augmented copies; point --data-root "
              "at the original (non-augmented) MRL images.")
    return list(records.values()), skipped, dupes, label_mismatch


def audit_existing_split(records):
    """Quantify subject overlap in the folder split the images came in."""
    by_split = defaultdict(set)
    n_images = Counter()
    for r in records:
        if r["folder_split"]:
            by_split[r["folder_split"]].add(r["subject"])
            n_images[r["folder_split"]] += 1
    if not by_split:
        return None
    train_subj = by_split.get("train", set())
    report = {"images_per_split": dict(n_images),
              "subjects_per_split": {k: len(v) for k, v in by_split.items()}}
    for split in ("val", "test"):
        if split not in by_split:
            continue
        shared = by_split[split] & train_subj
        imgs = [r for r in records if r["folder_split"] == split]
        seen = sum(r["subject"] in train_subj for r in imgs)
        report[f"{split}_subjects_also_in_train"] = len(shared)
        report[f"{split}_images_of_subjects_seen_in_train"] = seen
        report[f"{split}_images_total"] = len(imgs)
        report[f"{split}_fraction_images_seen_subjects"] = (
            round(seen / len(imgs), 4) if imgs else None)
    return report


def best_assignment(subject_counts, fractions, seed, trials):
    """Search random subject orders; keep the split closest to the targets."""
    subjects = sorted(subject_counts)
    total = sum(c["n"] for c in subject_counts.values())
    closed_ratio = sum(c["closed"] for c in subject_counts.values()) / total
    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(trials):
        order = subjects[:]
        rng.shuffle(order)
        # Fill test, then val, then train, by cumulative image count.
        groups = {"test": [], "val": [], "train": []}
        targets = {"test": fractions[2] * total, "val": fractions[1] * total}
        acc = {"test": 0, "val": 0}
        for s in order:
            n = subject_counts[s]["n"]
            for g in ("test", "val"):
                if acc[g] < targets[g] and acc[g] + n / 2 <= targets[g] * 1.25:
                    groups[g].append(s)
                    acc[g] += n
                    break
            else:
                groups["train"].append(s)
        if not all(groups.values()):
            continue
        score = 0.0
        for g, frac in zip(("train", "val", "test"), fractions):
            n = sum(subject_counts[s]["n"] for s in groups[g])
            c = sum(subject_counts[s]["closed"] for s in groups[g])
            if c == 0 or c == n:          # partition must contain both classes
                score = float("inf")
                break
            score += abs(n / total - frac) + 0.5 * abs(c / n - closed_ratio)
        if score < best_score:
            best, best_score = {g: sorted(v) for g, v in groups.items()}, score
    if best is None:
        raise RuntimeError("No valid subject assignment found; increase --trials")
    return best, best_score


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, skipped, dupes, mismatch = scan(data_root)
    if not records:
        raise SystemExit(
            f"No MRL-style file names (sXXXX_XXXXX_...) found under {data_root}. "
            "If the images were renamed, use the original MRL download, whose "
            "file names encode the subject ID.")
    subjects = Counter(r["subject"] for r in records)
    labels = Counter(r["label"] for r in records)
    print(f"Images: {len(records):,} | subjects: {len(subjects)} | "
          f"open: {labels['open']:,} | closed: {labels['closed']:,}")
    print(f"Skipped (unparsable names): {len(skipped):,} | "
          f"duplicate names ignored: {dupes:,} | folder/name label mismatches: {mismatch:,}")

    audit = audit_existing_split(records)
    if audit:
        (out_dir / "kaggle_split_audit.json").write_text(json.dumps(audit, indent=2))
        print("\nExisting folder split:")
        for k, v in audit.items():
            print(f"  {k}: {v}")
        if args.write_kaggle_manifest:
            kag = out_dir / "kaggle_split.csv"
            with kag.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["path", "subject", "label", "split"])
                for r in sorted(records, key=lambda r: (r["folder_split"] or "", r["path"])):
                    w.writerow([r["path"], r["subject"], r["label"], r["folder_split"]])
            print(f"Wrote {kag}")
    if args.audit_only:
        return

    counts = defaultdict(lambda: {"n": 0, "closed": 0})
    for r in records:
        counts[r["subject"]]["n"] += 1
        counts[r["subject"]]["closed"] += r["label"] == "closed"

    for seed in args.seeds:
        groups, score = best_assignment(counts, args.fractions, seed, args.trials)
        split_of = {s: g for g, members in groups.items() for s in members}
        manifest = out_dir / f"split_seed{seed}.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "subject", "label", "split"])
            for r in sorted(records, key=lambda r: (split_of[r["subject"]], r["path"])):
                w.writerow([r["path"], r["subject"], r["label"], split_of[r["subject"]]])
        summary = {"seed": seed, "score": round(score, 5), "subjects": groups}
        print(f"\nSeed {seed} (score {score:.4f}) -> {manifest}")
        for g in ("train", "val", "test"):
            n = sum(counts[s]["n"] for s in groups[g])
            c = sum(counts[s]["closed"] for s in groups[g])
            summary[f"{g}_images"] = n
            summary[f"{g}_closed"] = c
            print(f"  {g:5s}: {len(groups[g]):2d} subjects | {n:6,} images "
                  f"({n / len(records):.1%}) | closed {c / n:.1%}")
        (out_dir / f"split_seed{seed}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
