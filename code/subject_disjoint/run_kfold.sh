#!/bin/bash
# Grouped 5-fold subject-disjoint cross-validation (every subject tested once).
# Same settings as the three subject-disjoint runs in the paper (RUN_ON_3050TI.bat),
# seed fixed at 42 so that fold-to-fold variation reflects only the subjects.
cd "/b/ViT Model" || exit 1
LOG=subject_disjoint/runs/kfold5_run.log
for k in 0 1 2 3 4; do
  echo "=== fold $k start $(date)" >> "$LOG"
  venv/Scripts/python.exe subject_disjoint/train_subject_disjoint.py \
    --data-root data --manifest subject_disjoint/manifests/kfold5_fold$k.csv \
    --model vit-base --train-set base --batch-size 8 --grad-accum 4 \
    --epochs 10 --patience 3 --lr 2e-5 --seed 42 --resume >> "$LOG" 2>&1
  echo "=== fold $k exit $? $(date)" >> "$LOG"
done
echo "KFOLD_ALL_DONE $(date)" >> "$LOG"
