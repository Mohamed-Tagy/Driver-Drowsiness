# Driver Drowsiness: ViT Eye-State Classification with an Arduino Alert Prototype

Code, logs, results and model for the paper:

> M. W. Tagy, K. I. Shaheen, A. M. H. ElShazly, A. A. M. Takrony, H. ElSayed,
> D. Alkamr, M. Ashraf, and A. A. H. Ahmed, *Fine-Tuned Vision Transformer for
> Open/Closed Eye-State Classification with a Low-Cost Arduino Alert
> Prototype: A Replication and System-Integration Study*, Egypt-Japan
> University of Science and Technology, 2026.

A ViT-Base model (`google/vit-base-patch16-224`) is fine-tuned on the MRL Eye
Dataset to classify eyes as open or closed. The classifier drives a webcam
prototype (Haar-cascade face detection, a closed-eye counter, and an Arduino
Uno LED/buzzer module).

## Key results

| Evaluation | Test accuracy | Closed eyes classified as open |
|---|---|---|
| Original image-level split (selected model, Exp. 3) | 99.23% (95% CI 99.09–99.35) | 0.93% |
| Original split, control run of the new script | 99.19% | 0.83% |
| **Subject-disjoint, unseen people (3 splits)** | **98.41 ± 0.65%** (97.71–98.99) | **1.81 ± 1.00%** |

In the original split, every test subject also appears in training. On people
not seen during training, accuracy is lower and varies by person (92.9–100%).
These are static eye-state results, not validated drowsiness detection.

## Model weights

`checkpoint/` contains the model configuration and image-processor settings.
The weights are too large for a normal Git file and are attached to the
[Releases page](https://github.com/Mohamed-Tagy/Driver-Drowsiness/releases)
as `model_weights.zip` (318 MB; SHA-256
`bd7b8586c0b5e142790df845bd3267d64aa4b2b772baf9f309010e0d62a88bf2`).
Unzip it into `checkpoint/`; the extracted `model.safetensors` is listed in
`SHA256SUMS.txt`. Then load the model:

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
processor = AutoImageProcessor.from_pretrained("checkpoint")
model = AutoModelForImageClassification.from_pretrained("checkpoint")
# id2label: 0 = awake (open eyes), 1 = sleepy (closed eyes)
```

## Contents

| Folder | What | Paper section |
|---|---|---|
| `code/` | `train.py` (Hugging Face Trainer pipeline, Experiment 2), `augmentation.py` (offline 6× augmentation), `demo.py` (real-time prototype), `requirements.txt`, Arduino firmware | Materials and Methods |
| `code/subject_disjoint/` | Subject-disjoint split, training and aggregation scripts | Subject-Disjoint Evaluation |
| `subject_disjoint_runs/` | Control run (original split) and three subject-disjoint runs: configs, per-epoch history, logs, per-image predictions, metrics; `summary.md` | Subject-Disjoint Evaluation |
| `code/video_eval/` | Recording protocol, pipeline runner and event-level scorer for system-level evaluation | Conclusion (next steps) |
| `logs/` | Complete training logs of the three ViT experiments and the two demo sessions whose frame rates are quoted | ViT experiments; Prototype Operation |
| `results/` | Confusion matrices, training curves and GPU monitor of the selected run | Selected-model results |
| `checkpoint/` | Selected ViT-Base checkpoint (epoch 12): configuration files here, weights in the Release | Selected-model results |
| `manifests/` | Per-image train/val/test assignment for the original split and the three subject-disjoint splits; subject-overlap audit | Partitioning; Subject-Disjoint Evaluation |
| `ENVIRONMENT.md` | Hardware and software versions | Training Configuration |
| `SHA256SUMS.txt` | Checksums of every file, including the released weights | |

## Experiment-to-log mapping

| Paper | Model | Training data | Log |
|---|---|---|---|
| Exp. 1 | ViT-Tiny (`WinKawaks/vit-tiny-patch16-224`) | MRL | `logs/training_20260331_213932.log` |
| Exp. 2 | ViT-Base (`google/vit-base-patch16-224`) | MRL | `logs/training_combined_20260412_120715.log` (file name from the shared training script; data directory `data/MRL`) |
| Exp. 3 | ViT-Base | MRL Augmented | `logs/training_final_20260419_004212.log` |
| Subject-disjoint + control | ViT-Base | MRL | `subject_disjoint_runs/full_run.log` and the per-run folders |

## Data

The MRL Eye Dataset (Media Research Lab, VSB – Technical University of
Ostrava, http://mrl.cs.vsb.cz/eyedataset) is not redistributed here. The
experiments used its pre-split Kaggle redistribution
(https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset), which ships
`train/`, `val/` and `test/` folders (per class: 20% test, then 75/25
train/val, seed 42). `manifests/kaggle_split.csv` lists the partition of every
image. File names keep the MRL format `s0001_00014_0_0_0_0_0_01.png`, where
`s0001` is the subject.

## Reproducing the subject-disjoint evaluation

```bash
python code/subject_disjoint/make_subject_split.py --data-root data --seeds 42 43 44 --write-kaggle-manifest
python code/subject_disjoint/train_subject_disjoint.py --data-root data --manifest manifests/split_seed42.csv --model vit-base --train-set base --batch-size 8 --grad-accum 4 --epochs 10 --patience 3 --seed 42
python code/subject_disjoint/aggregate_results.py --runs subject_disjoint/runs
```

See `code/subject_disjoint/README.md` for details. Add
`--allow-subject-overlap` only for the control run on `manifests/kaggle_split.csv`.

## Known gaps

- The script that produced the Experiment 3 log (fused AdamW, BF16, cosine
  schedule) was not preserved; its settings are recorded in the log. The
  subject-disjoint evaluation uses a new script,
  `code/subject_disjoint/train_subject_disjoint.py`, validated by a control
  run on the original split (99.19% vs 99.21% for Experiment 2).
- The list of corrupted MRL files removed before training was not preserved.
- `augmentation.py` does not seed its two random variants, so the exact
  augmented images cannot be regenerated; `train_subject_disjoint.py` seeds
  them.
- No logs exist for two of the ResNet50 runs reported by a sub-team
  (57.75% and 99.03%); the CNN baseline code is kept by the sub-teams.
