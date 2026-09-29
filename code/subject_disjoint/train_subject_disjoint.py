"""
Train and evaluate a ViT eye-state classifier on a subject-disjoint split.

Settings default to those recorded for the paper's selected run
(logs/training_final_20260419_004212.log): google/vit-base-patch16-224,
AdamW (fused) lr 2e-5, weight decay 0.01, cosine schedule with 10% warmup,
gradient clipping 1.0, label smoothing 0.1, batch 32, 15 epochs, BF16,
checkpoint selected by validation accuracy.

Offline augmentation ("--train-set aug") reproduces the five variants of
augmentation.py on the fly: every training image appears once as the
original and once per variant in each epoch (6x, like MRL_AUGMENTED).
Unlike augmentation.py, the random variants are seeded per image, so a
run can be repeated exactly.

Examples:
    # 1-minute pipeline check, no model download (CPU is fine)
    python train_subject_disjoint.py --data-root data/MRL \
        --manifest subject_disjoint/manifests/split_seed42.csv --smoke

    # Full run (about 9 h on an RTX 5050 with --train-set aug)
    python train_subject_disjoint.py --data-root data/MRL \
        --manifest subject_disjoint/manifests/split_seed42.csv \
        --model vit-base --train-set aug --seed 42

Outputs (in --out-dir, default subject_disjoint/runs/<model>_<set>_seed<S>):
    train.log, history.json, best.pt, last.pt (for --resume),
    predictions_val.csv, predictions_test.csv, metrics.json
"""

import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

MODELS = {
    # name: (Hugging Face repo, revision recorded in the paper's training logs)
    "vit-base": ("google/vit-base-patch16-224",
                 "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"),
    "vit-tiny": ("WinKawaks/vit-tiny-patch16-224",
                 "77d1485af66b34d4ed0fe95dbb0c60c7496f950b"),
}
LABELS = ["open", "closed"]                 # index 0 = open, 1 = closed
IMG_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--model", choices=sorted(MODELS), default="vit-base")
    p.add_argument("--train-set", choices=["base", "aug"], default="aug",
                   help="aug = original + 5 offline-augmentation variants per image")
    p.add_argument("--online-aug", choices=["strong", "none"], default="strong",
                   help="On-the-fly transforms; 'strong' = train.py pipeline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"],
                   default="auto")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true",
                   help="Continue from last.pt in --out-dir")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny random model, 64/32/32 images, 1 epoch, no download")
    p.add_argument("--allow-subject-overlap", action="store_true",
                   help="Accept a manifest whose partitions share subjects "
                        "(only for the image-level Kaggle control run)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Offline-augmentation variants (parameters copied from augmentation.py)
# ---------------------------------------------------------------------------
def apply_variant(img, variant, rng, nprng):
    if variant == 1:                                   # horizontal flip
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    if variant == 2:                                   # rotate +20 (clockwise)
        return img.rotate(-20, resample=Image.BILINEAR, expand=False,
                          fillcolor=(0, 0, 0))
    if variant == 3:                                   # rotate -20
        return img.rotate(20, resample=Image.BILINEAR, expand=False,
                          fillcolor=(0, 0, 0))
    if variant == 4:                                   # brightness/contrast preset
        choice = rng.choice(["bright", "dark", "high_contrast"])
        if choice == "bright":
            img = ImageEnhance.Brightness(img).enhance(1.4)
            return ImageEnhance.Contrast(img).enhance(1.2)
        if choice == "dark":
            img = ImageEnhance.Brightness(img).enhance(0.6)
            return ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        return ImageEnhance.Brightness(img).enhance(1.1)
    if variant == 5:                                   # combined transform
        if rng.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rng.uniform(-10, 10), resample=Image.BILINEAR,
                         fillcolor=(0, 0, 0))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.85, 1.15))
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))
        arr = np.asarray(img).astype(np.float32)
        arr = arr + nprng.normal(0, rng.uniform(3, 12), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        w, h = img.size
        r = rng.uniform(0.88, 0.96)
        cw, ch = int(w * r), int(h * r)
        left, top = rng.randint(0, w - cw), rng.randint(0, h - ch)
        return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)
    return img


class MRLDataset(Dataset):
    def __init__(self, rows, data_root, transform, n_variants=1, seed=0):
        self.rows, self.root = rows, Path(data_root)
        self.transform, self.n_variants, self.seed = transform, n_variants, seed

    def __len__(self):
        return len(self.rows) * self.n_variants

    def __getitem__(self, i):
        idx, variant = divmod(i, self.n_variants)
        row = self.rows[idx]
        with Image.open(self.root / row["path"]) as im:
            img = im.convert("RGB")
        if variant:
            # Same variant image in every epoch, as with the offline folder.
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            rng = random.Random(f"{self.seed}-{idx}-{variant}")
            nprng = np.random.default_rng([self.seed, idx, variant])
            img = apply_variant(img, variant, rng, nprng)
        return self.transform(img), LABELS.index(row["label"]), idx


def build_transforms(mean, std, online):
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    evaluation = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE))] + norm)
    if online == "none":
        return evaluation, evaluation
    train = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        *norm,
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    return train, evaluation


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(args):
    from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                              ViTConfig, ViTForImageClassification)
    id2label = dict(enumerate(LABELS))
    label2id = {v: k for k, v in id2label.items()}
    if args.smoke:
        cfg = ViTConfig(image_size=IMG_SIZE, patch_size=16, hidden_size=64,
                        num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=128, num_labels=2,
                        id2label=id2label, label2id=label2id)
        return ViTForImageClassification(cfg), [0.5] * 3, [0.5] * 3, "smoke-random"
    repo, rev = MODELS[args.model]
    processor = AutoImageProcessor.from_pretrained(repo, revision=rev)
    model = AutoModelForImageClassification.from_pretrained(
        repo, revision=rev, num_labels=2, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)
    return model, list(processor.image_mean), list(processor.image_std), f"{repo}@{rev}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 3), round(100 * (c + h), 3)]


def compute_metrics(labels, preds, subjects):
    cm = [[0, 0], [0, 0]]
    for y, p in zip(labels, preds):
        cm[y][p] += 1
    n = sum(map(sum, cm))
    correct = cm[0][0] + cm[1][1]
    per_class = {}
    for c, name in enumerate(LABELS):
        tp = cm[c][c]
        prec = tp / max(1, cm[0][c] + cm[1][c])
        rec = tp / max(1, sum(cm[c]))
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        per_class[name] = {"precision": round(100 * prec, 3),
                           "recall": round(100 * rec, 3),
                           "f1": round(100 * f1, 3), "support": sum(cm[c])}
    by_subject = defaultdict(lambda: [0, 0])
    for y, p, s in zip(labels, preds, subjects):
        by_subject[s][0] += int(y == p)
        by_subject[s][1] += 1
    return {
        "n": n,
        "accuracy": round(100 * correct / n, 3),
        "accuracy_wilson95": wilson(correct, n),
        "macro_f1": round(sum(v["f1"] for v in per_class.values()) / 2, 3),
        "closed_as_open_rate": round(100 * cm[1][0] / max(1, sum(cm[1])), 3),
        "open_as_closed_rate": round(100 * cm[0][1] / max(1, sum(cm[0])), 3),
        "confusion_matrix_rows_actual_open_closed": cm,
        "per_class": per_class,
        "per_subject_accuracy": {s: round(100 * c / t, 2)
                                 for s, (c, t) in sorted(by_subject.items())},
    }


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, loss_fn):
    model.eval()
    total_loss, idxs, labels, preds, probs = 0.0, [], [], [], []
    for x, y, idx in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(pixel_values=x).logits
        logits = logits.float()
        total_loss += loss_fn(logits, y).item() * len(y)
        p = torch.softmax(logits, 1)[:, 1]
        idxs += idx.tolist()
        labels += y.tolist()
        preds += logits.argmax(1).tolist()
        probs += p.tolist()
    return total_loss / max(1, len(labels)), idxs, labels, preds, probs


def write_predictions(path, rows, idxs, labels, preds, probs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "subject", "label", "pred", "prob_closed"])
        for i, y, p, q in zip(idxs, labels, preds, probs):
            w.writerow([rows[i]["path"], rows[i]["subject"], LABELS[y], LABELS[p],
                        f"{q:.6f}"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.smoke:
        args.epochs, args.batch_size, args.workers = 1, 8, 0
    split_name = Path(args.manifest).stem          # e.g. kaggle_split, split_seed42
    run_name = (f"smoke_seed{args.seed}" if args.smoke else
                f"{split_name}_{args.model}_{args.train_set}_{args.online_aug}"
                f"_seed{args.seed}")
    out = Path(args.out_dir or Path("subject_disjoint/runs") / run_name)
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(out / "train.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("train")
    seed_everything(args.seed)

    # ---- data ---------------------------------------------------------------
    with open(args.manifest, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rows = {s: [r for r in all_rows if r["split"] == s] for s in ("train", "val", "test")}
    subj = {s: {r["subject"] for r in v} for s, v in rows.items()}
    overlap = (subj["train"] & subj["test"]) | (subj["train"] & subj["val"]) | \
              (subj["val"] & subj["test"])
    if overlap and not args.allow_subject_overlap:
        raise SystemExit(f"Manifest is not subject-disjoint: {sorted(overlap)}")
    if args.smoke:
        rng = random.Random(args.seed)
        rows = {s: rng.sample(v, min(len(v), n))
                for (s, v), n in zip(rows.items(), (64, 32, 32))}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.precision == "fp32" or device.type != "cuda":
        amp_dtype = None
    elif args.precision == "fp16":
        amp_dtype = torch.float16
    elif args.precision == "bf16" or torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    model, mean, std, model_id = build_model(args)
    model.to(device)
    train_tf, eval_tf = build_transforms(mean, std, args.online_aug)
    n_variants = 6 if args.train_set == "aug" else 1
    ds = {"train": MRLDataset(rows["train"], args.data_root, train_tf, n_variants, args.seed),
          "val": MRLDataset(rows["val"], args.data_root, eval_tf),
          "test": MRLDataset(rows["test"], args.data_root, eval_tf)}
    common = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                  worker_init_fn=worker_init, persistent_workers=args.workers > 0)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              generator=gen, drop_last=False, **common)
    val_loader = DataLoader(ds["val"], batch_size=args.batch_size * 2, **common)
    test_loader = DataLoader(ds["test"], batch_size=args.batch_size * 2, **common)

    # ---- optimisation ------------------------------------------------------
    import transformers
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay,
                            fused=device.type == "cuda")
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_factor(step):                   # linear warmup, then cosine decay to 0
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    config = {**vars(args), "model_id": model_id, "n_variants": n_variants,
              "amp_dtype": str(amp_dtype), "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "python": platform.python_version(),
              "total_steps": total_steps, "warmup_steps": warmup_steps,
              "subjects": {s: sorted(v) for s, v in subj.items()},
              "images": {s: len(v) for s, v in rows.items()},
              "train_samples_per_epoch": len(ds["train"])}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    log.info("Config: %s", json.dumps({k: v for k, v in config.items()
                                        if k != "subjects"}))
    log.info("Subjects train/val/test: %d/%d/%d", *(len(subj[s]) for s in
                                                   ("train", "val", "test")))

    history, best_acc, start_epoch, bad_epochs = [], -1.0, 1, 0
    last_ckpt = out / "last.pt"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        history, best_acc = ck["history"], ck["best_acc"]
        start_epoch, bad_epochs = ck["epoch"] + 1, ck["bad_epochs"]
        log.info("Resumed after epoch %d (best val acc %.2f%%)", ck["epoch"], best_acc)

    # ---- training ----------------------------------------------------------
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        gen.manual_seed(args.seed * 1000 + epoch)    # same order after --resume
        torch.manual_seed(args.seed * 1000 + epoch)  # online-augmentation draws
        model.train()
        t0, run_loss, run_correct, seen = time.time(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for step, (x, y, _) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                logits = model(pixel_values=x).logits
            loss = loss_fn(logits.float(), y)
            scaler.scale(loss / args.grad_accum).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
            if step % 500 == 0:
                log.info("  epoch %d step %d/%d loss %.4f acc %.2f%%", epoch, step,
                         len(train_loader), run_loss / seen, 100 * run_correct / seen)

        val_loss, _, vy, vp, _ = evaluate(model, val_loader, device, amp_dtype, loss_fn)
        val_acc = 100 * sum(a == b for a, b in zip(vy, vp)) / len(vy)
        rec = {"epoch": epoch, "train_loss": run_loss / seen,
               "train_acc": 100 * run_correct / seen, "val_loss": val_loss,
               "val_acc": val_acc, "minutes": (time.time() - t0) / 60}
        history.append(rec)
        improved = val_acc > best_acc
        if improved:
            best_acc, bad_epochs = val_acc, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad_epochs += 1
        log.info("Epoch %d: train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%%%s "
                 "(%.1f min)", epoch, rec["train_loss"], rec["train_acc"], val_loss,
                 val_acc, "  <- best" if improved else "", rec["minutes"])
        (out / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "history": history, "best_acc": best_acc, "epoch": epoch,
                    "bad_epochs": bad_epochs}, last_ckpt)
        if bad_epochs >= args.patience:
            log.info("Early stopping after epoch %d", epoch)
            break

    # ---- final evaluation with the best checkpoint --------------------------
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    results = {"config": {k: v for k, v in config.items() if k != "subjects"},
               "subjects": config["subjects"],
               "best_epoch": max(history, key=lambda h: h["val_acc"])["epoch"],
               "training_minutes": round((time.time() - t_start) / 60, 1)}
    for split, loader in (("val", val_loader), ("test", test_loader)):
        loss, idxs, labels, preds, probs = evaluate(model, loader, device,
                                                    amp_dtype, loss_fn)
        subjects = [rows[split][i]["subject"] for i in idxs]
        results[split] = compute_metrics(labels, preds, subjects)
        results[split]["loss"] = round(loss, 5)
        write_predictions(out / f"predictions_{split}.csv", rows[split], idxs,
                          labels, preds, probs)
        m = results[split]
        log.info("%s: accuracy %.2f%% (95%% CI %s) | macro F1 %.2f%% | "
                 "closed->open %.2f%%", split.upper(), m["accuracy"],
                 m["accuracy_wilson95"], m["macro_f1"], m["closed_as_open_rate"])
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    log.info("Per-subject test accuracy: %s", results["test"]["per_subject_accuracy"])
    log.info("Done. Outputs in %s", out.resolve())


if __name__ == "__main__":
    main()
