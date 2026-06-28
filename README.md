---
title: CarSpec AI
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# CarSpec AI — Vehicle Multi-Attribute Recognition System

> Module 2 Project · Computer Vision · Based on the CompCars Dataset

## Project Introduction

CarSpec AI is a computer vision-based vehicle multi-attribute recognition system. Upload a vehicle exterior photo, and the system simultaneously predicts three attributes: **vehicle type**, **door count**, and **seat count**, and provides interpretable visual feature analysis.

## Core Approach

1. **Multi-task Joint Learning**: MobileNetV2 shared backbone + three classification heads, using task correlations to improve performance
2. **Interpretable Visual Features**: Extract 50+ handcrafted features (color histogram, HOG, texture, body proportions, symmetry), providing better interpretability than pure CNNs
3. **Three-Model Comparison**: Naive baseline / Classical ML (Random Forest) / Deep Learning (MobileNetV2 multi-task)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the training pipeline (optional — generates models and evaluation outputs)

```bash
python setup.py --sample-size 2000 --epochs 3
```

This produces `models/*.pkl`, `models/deep_multitask.pt`, and `data/outputs/metrics.json` + plots.

### 3. Start the web application

```bash
python main.py
```

App starts on `http://localhost:7860`. Models are auto-downloaded from HuggingFace Hub on first run.

### 4. Deploy (Docker)

```bash
docker build -t carspec-ai .
docker run -p 7860:7860 carspec-ai
```

## Online Demo

Visit the HuggingFace Space: https://hanfuzhao781-carspec-ai.hf.space

- `GET /health` → `{"status":"ok","models_loaded":7}`
- `POST /predict` with a vehicle image → JSON with `prediction`, `confidence`, `top_k`, `feedback`

## Project Structure

```
├── README.md
├── REPORT.md               <- Full project report
├── PITCH.md                <- 5-minute pitch script
├── GRADING.md              <- Rubric mapping
├── requirements.txt        <- Full deps (training + deploy)
├── requirements-deploy.txt <- Lightweight deploy deps
├── Makefile
├── setup.py                <- Training pipeline (9 steps)
├── main.py                 <- Flask web application
├── Dockerfile
├── scripts/
│   ├── data.py             <- Data loading
│   ├── make_dataset.py     <- Data download
│   ├── synthetic_data.py   <- Synthetic data generation
│   ├── features.py         <- Interpretable feature extraction
│   ├── model.py            <- Three model implementations
│   ├── experiment.py       <- Experiment framework (robustness, gating, head-tail)
│   └── eda.py              <- Exploratory data analysis
├── models/                 <- Trained models (downloaded from HF Hub at runtime)
├── data/
│   ├── raw/                <- Raw data (CompCars or synthetic)
│   ├── processed/          <- Processed splits + features
│   └── outputs/            <- metrics.json + plots + experiments
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── samples/            <- Sample images for demo
├── templates/index.html
└── .github/
    ├── workflows/keep-alive.yml
    └── pull_request_template.md
```

## Three Model Locations

| Model | Code Location | Trained Model | Description |
|------|---------|-----------|-------------|
| Naive Baseline | `scripts/model.py::NaiveBaseline` | `models/naive_*.pkl` | Majority-class classifier |
| Classical ML | `scripts/model.py::ClassicalModel` | `models/classical_*.pkl` | Random Forest + 50-dim interpretable features |
| Deep Learning | `scripts/model.py::DeepMultiTaskModel` | `models/deep_multitask.pt` | MobileNetV2 multi-task (car_type/door_count/seat_count) |

## Dataset

**CompCars** (CVPR 2015): 136,726 whole-vehicle images, annotated with vehicle type, door count, seat count, maximum speed, displacement, and other attributes.

- Official website: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
- Paper: Yang et al., "A Large-Scale Car Dataset for Fine-Grained Categorization and Verification"

When CompCars is unavailable, a synthetic dataset (6,000 simulated vehicle images) is generated automatically to validate the full pipeline end-to-end.

## Evaluation Outputs

After running `python setup.py`, the following artifacts are produced in `data/outputs/`:

- `metrics.json` — all metrics (naive_majority, naive_random, classical, deep, robustness, head_tail, error_cases, meta)
- `model_comparison.png` — bar chart comparing three models across tasks
- `confusion_matrix.png` + `confusion_matrix.npy` — aggregate confusion matrix
- `robustness.png` — corruption robustness (gaussian_noise / motion_blur / jpeg_compression / pixelate)
- `confidence_curve.png` — confidence gating accuracy vs coverage
- `confidence_analysis.json` — per-threshold accuracy/coverage rows
- `sample_images.png` — sample images per class
- `plots/cm_*.png` — per-task per-model confusion matrices
- `plots/eda_dist_*.png` — class distribution bar charts

## Originality Statement

This project is original work for the Module 2 Project of the Computer Vision course. The contributions are:

1. **Multi-task Joint Learning + Interpretable Feature Fusion**: Combines handcrafted interpretable features with deep features, improving performance while providing interpretability — distinct from prior single-task or pure-deep approaches.
2. **Attribute Correlation Modeling**: Uses the correlations between vehicle type, door count, and seat count through a shared MobileNetV2 backbone + three classification heads.
3. **End-to-End Interpretable System**: Not only predicts attributes but also generates natural language explanations (e.g., "dominant color is blue", "aspect ratio 1.2 → leans toward SUV").
4. **Insight-Driven Experiments**: Beyond accuracy maximization, the project conducts focused experiments on corruption robustness, confidence gating (selective prediction), and head/tail class analysis — providing practical deployment insights.

All code was authored by the project developer. No third-party code beyond standard library and framework usage (PyTorch, scikit-learn, Flask, etc.). The CompCars dataset is used under its non-commercial research license.

## Tech Stack

- **Deep Learning**: PyTorch, torchvision MobileNetV2
- **Classical ML**: scikit-learn, Random Forest
- **Feature Engineering**: scikit-image (HOG), LBP texture
- **Web Framework**: Flask, Gunicorn
- **Deployment**: Docker, HuggingFace Spaces
- **CI**: GitHub Actions (keep-alive)
