"""
Re-evaluate saved checkpoints on the original (Kaggle) test partition and
write per-image predictions, so that the image-level configurations can be
compared with paired tests and subject-level statistics.

    python subject_disjoint/evaluate_checkpoints.py --data-root data \
        --model exp1="B:/Demo 2/models/vit-drowsiness-enhanced" \
        --model exp3=models/vit-base-mrl-augmented-2/final_model

Preprocessing matches the evaluation transform of the training scripts:
RGB, resize to 224x224 (bilinear), scale to [0,1], normalize with the
checkpoint's image-processor mean/std. No training is performed.
Output: subject_disjoint/checkpoint_eval/<name>_test_predictions.csv
        (path, subject, label, pred, p_closed) and <name>_metrics.json.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

CLASSES = {"awake": 0, "sleepy": 1}      # open = 0, closed = 1


def load_images(root):
    items = []
    for cls, y in CLASSES.items():
        for p in sorted((root / "test" / cls).glob("*")):
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                items.append((p, y))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model", action="append", required=True, help="name=path")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="subject_disjoint/checkpoint_eval")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_images(Path(a.data_root))
    print(f"{len(items)} test images")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for spec in a.model:
        name, path = spec.split("=", 1)
        try:
            proc = AutoImageProcessor.from_pretrained(path)
        except OSError:    # Trainer checkpoints lack preprocessor_config.json
            proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            print(f"{name}: no image-processor config; using the base checkpoint's")
        model = AutoModelForImageClassification.from_pretrained(path).eval().to(dev)
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        closed_idx = next(i for i, l in id2label.items() if "sleep" in l or "clos" in l)
        mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(dev)
        std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(dev)
        rows = []
        with torch.no_grad():
            for i in range(0, len(items), a.batch_size):
                chunk = items[i:i + a.batch_size]
                x = np.stack([np.asarray(Image.open(p).convert("RGB")
                                         .resize((224, 224), Image.BILINEAR), np.float32)
                              for p, _ in chunk]) / 255.0
                x = (torch.from_numpy(x).permute(0, 3, 1, 2).to(dev) - mean) / std
                prob = torch.softmax(model(pixel_values=x).logits.float(), 1)[:, closed_idx]
                for (p, y), pc in zip(chunk, prob.cpu().tolist()):
                    rows.append((p.name, p.name.split("_")[0], y, int(pc > 0.5), pc))
        with open(out / f"{name}_test_predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "subject", "label", "pred", "p_closed"])
            w.writerows([(r[0], r[1], r[2], r[3], f"{r[4]:.6f}") for r in rows])
        y = np.array([r[2] for r in rows])
        pr = np.array([r[3] for r in rows])
        m = {"model": path, "n": len(rows), "accuracy": float((y == pr).mean()),
             "tn": int(((y == 0) & (pr == 0)).sum()), "fp": int(((y == 0) & (pr == 1)).sum()),
             "fn": int(((y == 1) & (pr == 0)).sum()), "tp": int(((y == 1) & (pr == 1)).sum()),
             "params": sum(p.numel() for p in model.parameters())}
        (out / f"{name}_metrics.json").write_text(json.dumps(m, indent=2))
        print(name, json.dumps(m))


if __name__ == "__main__":
    main()
