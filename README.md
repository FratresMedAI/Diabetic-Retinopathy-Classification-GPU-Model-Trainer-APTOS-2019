# APTOS-DR-GPU v4

GPU-first diabetic retinopathy classification notebook, polished for reproducibility, portfolio quality, and clean Git history.

> **Clinical Disclaimer:** Research use only — not for clinical diagnosis.

## Overview

This project trains a deep learning model to classify diabetic retinopathy severity from retinal fundus images using the APTOS 2019 Blindness Detection dataset.

The notebook is optimized for high-end GPU environments and is structured to be:

- **Reproducible**
- **Source-clean**
- **Employer-friendly**
- **Fast on modern CUDA hardware**
- **Focused on clinically meaningful metrics**

## What makes this version strong

- **GPU-first training path**
  - BF16 / AMP support on CUDA
  - TF32 enabled on supported hardware
  - `channels_last` memory format
  - `torch.compile` on GPU
  - Pinned memory and non-blocking tensor transfer

- **Severity-aware learning**
  - Class-balanced sampling
  - Weighted loss for imbalance
  - Balanced accuracy, macro F1, and per-class recall
  - Quadratic Weighted Kappa (QWK) as the main model selection metric

- **Professional execution flow**
  - Clean bootstrap cell
  - Kaggle dataset download and extraction
  - W&B experiment tracking
  - Hugging Face + Kaggle auth support
  - Versioned outputs and artifacts

- **Git-ready presentation**
  - Notebook outputs cleared before commit
  - Recruiter-friendly summary and quick-start information
  - Consistent version branding throughout

## Repository contents

- `APTOS_DR_GPU_v4.ipynb` — main training notebook
- `README.md` — project overview and usage guide

## Dataset

The notebook expects the **APTOS 2019 Blindness Detection** competition dataset from Kaggle.

Expected dataset layout after extraction:

- `train.csv`
- `train_images/`

The notebook can download and extract the dataset automatically if Kaggle credentials are available.

## Requirements

Recommended environment:

- Python 3.11
- PyTorch 2.8.x
- CUDA 12.8
- High-VRAM NVIDIA GPU (H100 / A100 preferred)

Key Python packages used by the notebook:

- `torch`
- `torchvision`
- `timm`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `opencv-python-headless`
- `Pillow`
- `kaggle`
- `wandb`
- `tqdm`
- `huggingface_hub`

## Quick start

1. Open `APTOS_DR_GPU_v4.ipynb`.
2. Run **Cell 0** to install/check dependencies.
3. Run the notebook top to bottom.
4. Confirm the dataset is downloaded and extracted.
5. Train, validate, and review the final OOF metrics and artifacts.

## Recommended runtime

For best performance, use a CUDA GPU environment with plenty of VRAM.

Suggested RunPod image:

```text
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
```

## Notebook workflow

- **Cell 0** — bootstrap, installs, auth helpers, shared imports
- **Cell 1** — environment, config, GPU setup, W&B init
- **Cell 2** — Kaggle download and fast extraction
- **Cell 3** — data cleaning, fold creation, path resolution
- **Cell 4** — dataset and dataloaders
- **Cell 5** — model, loss, optimizer, AMP setup
- **Cell 6** — training loop and checkpointing
- **Cell 7** — temperature scaling and TTA helper
- **Cell 8** — evaluation, QWK, accuracy, balanced accuracy, F1, recall
- **Cell 9** — plots and artifact manifest
- **Cell 10** — final summary and reproducibility checklist

## Metrics tracked

This notebook emphasizes the metrics that matter for an imbalanced, severity-ordered medical classification problem:

- Quadratic Weighted Kappa (QWK)
- Accuracy
- Balanced Accuracy
- Macro F1
- Per-class recall
- Temperature-calibrated inference

## Output artifacts

The notebook writes versioned outputs into:

- `results/`
- `artifacts/`
- `checkpoints/`

Typical artifacts include:

- Best model checkpoint
- Fold split CSV
- Training history CSV
- OOF prediction CSV
- Confusion matrix image
- Training curve plot
- Artifact manifest JSON

## Credentials and authentication

The notebook includes support for:

- Kaggle API authentication
- Weights & Biases tracking
- Hugging Face Hub authentication for pretrained model downloads

> **Security note:** Never commit real tokens to Git. Use environment variables or secure credential storage for public repos.

## Common troubleshooting

- **Kaggle auth errors**
  - Confirm `kaggle.json` is valid and in the expected location.
  - Verify Kaggle API access is enabled.

- **Slow data extraction**
  - The notebook uses `7z` when available for fast extraction.
  - If `7z` is missing, Cell 0 attempts to install it.

- **GPU usage looks low**
  - Increase batch size if VRAM allows.
  - Make sure you are running on CUDA hardware.
  - Check that `torch.compile`, AMP, and `channels_last` are enabled.

- **W&B login issues**
  - Set `WANDB_MODE=offline` if needed.

- **OOM / memory pressure**
  - Lower `BATCH_SIZE` or `IMG_SIZE`.
  - Reduce `EFFECTIVE_BATCH_TARGET`.

## Reproducibility notes

- Fixed random seeds
- Versioned outputs
- Config captured in checkpoint and W&B
- Source-clean notebook committed for Git

## Project goals

This notebook is designed to show more than raw model accuracy. It demonstrates:

- practical GPU engineering
- severity-aware medical ML training
- clean experimentation discipline
- readable, professional project presentation

## License

No explicit license has been added yet. If you plan to share this publicly, add a license file that matches your intended use.

## Acknowledgements

- Kaggle APTOS 2019 Blindness Detection competition
- PyTorch
- timm
- Weights & Biases
- Hugging Face

---

If you want, I can also create a matching **`requirements.txt`** and a short **project tree** section so the repo looks even more polished before you push.
