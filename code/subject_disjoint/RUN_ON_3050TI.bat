@echo off
REM ==================================================================
REM Subject-disjoint study on the RTX 3050 Ti Laptop GPU (4 GB).
REM Settings follow Experiment 2 of the paper, which ran on this GPU:
REM ViT-Base, original MRL training set, batch 8 x 4 accumulation
REM (effective 32), lr 2e-5, 10 epochs, early-stopping patience 3.
REM   1. control run on the Kaggle split  (should give ~99.2%)
REM   2. three subject-disjoint splits    (seeds 42, 43, 44)
REM Each run takes roughly 3-4 hours. Re-running this file resumes.
REM ==================================================================
cd /d "%~dp0\.."
set PY=venv\Scripts\python.exe
set COMMON=--data-root data --model vit-base --train-set base --batch-size 8 --grad-accum 4 --epochs 10 --patience 3 --lr 2e-5 --resume

%PY% subject_disjoint\train_subject_disjoint.py --data-root data --manifest subject_disjoint\manifests\split_seed42.csv --smoke
if errorlevel 1 goto :error

%PY% subject_disjoint\train_subject_disjoint.py %COMMON% --manifest subject_disjoint\manifests\kaggle_split.csv --seed 42 --allow-subject-overlap
if errorlevel 1 goto :error

for %%S in (42 43 44) do (
    %PY% subject_disjoint\train_subject_disjoint.py %COMMON% --manifest subject_disjoint\manifests\split_seed%%S.csv --seed %%S
    if errorlevel 1 goto :error
)

%PY% subject_disjoint\aggregate_results.py
echo.
echo Done. Results: subject_disjoint\runs\summary.md
pause
exit /b 0

:error
echo.
echo A step failed; see the messages above. Running this file again resumes finished work.
pause
exit /b 1
