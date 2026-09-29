# Environment

| | Experiments 1-2 | Experiment 3 (selected model) |
|---|---|---|
| GPU | NVIDIA RTX 3050 Ti Laptop GPU, 4.3 GB | NVIDIA RTX 5050, 8.5 GB |
| CUDA | 12.1 | 12.8 |
| PyTorch | not recorded in the log | 2.12.0.dev20260408+cu128 (nightly; required for Blackwell) |
| Transformers | not recorded in the log | 5.5.4 (recorded in `checkpoint/config.json`) |
| Precision | FP16 mixed precision | BF16 with TF32 (FP16 produced NaN losses) |
| Checkpoint revision | ViT-Base: `google/vit-base-patch16-224@3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3`; ViT-Tiny: `WinKawaks/vit-tiny-patch16-224@77d1485af66b34d4ed0fe95dbb0c60c7496f950b` | same ViT-Base revision |

Minimum package versions: `code/requirements.txt`. Install PyTorch separately
for your CUDA version (https://pytorch.org).
