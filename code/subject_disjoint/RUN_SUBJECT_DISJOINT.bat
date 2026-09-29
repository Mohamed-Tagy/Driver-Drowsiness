@echo off
REM Subject-disjoint study: 3 subject splits x ViT-Base (augmented training set)
REM Run from anywhere; expects the MRL images in data\MRL and the project venv.
cd /d "%~dp0\.."
call venv\Scripts\activate.bat

python subject_disjoint\make_subject_split.py --data-root data\MRL --seeds 42 43 44
if errorlevel 1 goto :error

python subject_disjoint\train_subject_disjoint.py --data-root data\MRL --manifest subject_disjoint\manifests\split_seed42.csv --smoke
if errorlevel 1 goto :error

for %%S in (42 43 44) do (
    python subject_disjoint\train_subject_disjoint.py --data-root data\MRL --manifest subject_disjoint\manifests\split_seed%%S.csv --model vit-base --train-set aug --seed %%S --resume
    if errorlevel 1 goto :error
)

python subject_disjoint\aggregate_results.py
echo.
echo Done. See subject_disjoint\runs\summary.md
pause
exit /b 0

:error
echo.
echo A step failed; see the messages above. Re-running this file resumes finished work.
pause
exit /b 1
