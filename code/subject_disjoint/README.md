# Subject-disjoint evaluation

This folder answers the instructor's main request: accuracy measured on
**people the model has never seen**. The original experiments used the
Kaggle train/val/test folders, in which the same 37 MRL subjects appear in
every partition.

## Files

| File | Purpose | Needs |
|---|---|---|
| `make_subject_split.py` | Splits the 37 subjects into train/val/test groups, one split per seed; writes the manifests; audits the Kaggle split for subject overlap | Python only |
| `train_subject_disjoint.py` | Trains and evaluates a ViT on one manifest, with the settings of the paper's selected run | PyTorch, torchvision, transformers, GPU |
| `aggregate_results.py` | Mean ± SD across seeds; exact McNemar tests between configurations; LaTeX rows | Python only |
| `RUN_SUBJECT_DISJOINT.bat` | Runs the whole study (3 seeds) | venv from `quick_install.bat` |

## Steps

Run all commands from the project root (`B:\ViT Model`) with the project's
venv activated.

**1. Put the MRL images in `data\MRL`.** The Kaggle folders
(`train\awake`, `train\sleepy`, `val\...`, `test\...`) are fine. Use the
original images, not `MRL_AUGMENTED`. File names must keep the MRL format
`s0001_00014_0_0_0_0_0_01.png`. The script stops with a message if they
don't; in that case, download the original dataset from
http://mrl.cs.vsb.cz/eyedataset.

**2. Audit the Kaggle split and create the subject-disjoint splits:**

```
python subject_disjoint\make_subject_split.py --data-root data\MRL --seeds 42 43 44
```

This prints how many test images belong to subjects that also appear in
Kaggle's training folder, a number the paper can quote (it is saved to
`manifests\kaggle_split_audit.json`). It then prints, for each seed, which
subjects form each partition and the image and class balance.

**3. Check the pipeline in about a minute** (tiny random model, 128 images,
no download):

```
python subject_disjoint\train_subject_disjoint.py --data-root data\MRL --manifest subject_disjoint\manifests\split_seed42.csv --smoke
```

**4. Run the study:** `RUN_SUBJECT_DISJOINT.bat`, or one run at a time:

```
python subject_disjoint\train_subject_disjoint.py --data-root data\MRL --manifest subject_disjoint\manifests\split_seed42.csv --model vit-base --train-set aug --seed 42
```

If a run is interrupted, repeat the same command with `--resume`.

**5. Summarize:**

```
python subject_disjoint\aggregate_results.py
```

## Time estimates (RTX 5050, from the paper's final log)

| Configuration | Per run | 3 seeds |
|---|---|---|
| ViT-Base, augmented training set (`--train-set aug`) | ~9 h | ~27 h |
| ViT-Base, original training set (`--train-set base`) | ~1.5 h | ~4.5 h |
| ViT-Tiny, original training set | ~0.5 h | ~1.5 h |

Subject-disjoint training sets are a similar size to the Kaggle one (about
60% of the images), so the times are comparable.

## What each run saves (`subject_disjoint\runs\<name>\`)

- `config.json`: all settings, library versions, GPU, and the subject lists.
- `train.log` and `history.json`: per-epoch losses and accuracies.
- `best.pt`: best checkpoint by validation accuracy; `last.pt` is used by `--resume`.
- `predictions_val.csv` and `predictions_test.csv`: one row per image (needed for McNemar tests).
- `metrics.json`: accuracy with a 95% Wilson interval, per-class P/R/F1, confusion matrix, closed→open rate, and **accuracy per test subject**.

## Differences from the original pipeline (state these in the paper)

- **Split:** by subject, not by image; several seeds give several different subject splits.
- **Offline augmentation:** the five variants of `augmentation.py` are generated on the fly with a per-image seed rather than written to disk as JPEG. The random variants are therefore reproducible, and no JPEG re-compression is applied.
- **Online augmentation:** `--online-aug strong` (default) is the `train.py` pipeline. The final run's log does not record whether it was used; if you know it wasn't, use `--online-aug none`.
