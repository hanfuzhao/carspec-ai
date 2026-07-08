---
title: CarSpec AI
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# CarSpec AI - Vehicle Multi-Attribute Recognition

> Module 2 Project · Computer Vision · CompCars Dataset

## What it does

CarSpec AI reads a car photo and returns type, door count, and seat count. Upload a vehicle exterior photo, get back three attributes plus a breakdown of the visual features behind each call.

## How it works

1. **Multi-task learning**: MobileNetV2 shared backbone + three classification heads, so the three tasks teach each other
2. **Interpretable features**: 50 handcrafted features (color histogram, HOG, texture, body proportions, symmetry) sit alongside the deep features - more interpretable than a pure CNN
3. **Three-model comparison**: Naive baseline / Classical (Random Forest) / Deep (MobileNetV2 multi-task)

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the training pipeline (optional - generates models and evaluation outputs)

Two paths are available:

**Full synthetic pipeline (small-scale demo):**
```bash
python setup.py --sample-size 2000 --epochs 3
```

**Large-scale real-data training (used for final results):**
```bash
python scripts/download_large_dataset.py   # crawl 5000+ photos from Bing
python scripts/clean_dataset.py            # dedup + validate + aspect filter
python scripts/train_large.py              # train RF + MobileNetV2 on 4869 images (MPS)
python scripts/eval_large.py               # re-evaluate + regenerate metrics.json
```

Both paths produce `models/*.pkl`, `models/deep_multitask.pt`, and `data/outputs/metrics.json` + plots.

### 3. Start the web application

```bash
python main.py
```

App starts on `http://localhost:5000` (default). For Docker/HF Space, set `PORT=7860` env var. Models are auto-downloaded from HuggingFace Hub on first run.

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
├── TECHNICAL_REPORT.md     <- Full project report
├── GRADING.md              <- Rubric mapping
├── requirements.txt        <- Full deps (training + deploy)
├── requirements-deploy.txt <- Lightweight deploy deps
├── Makefile
├── setup.py                <- Training pipeline (synthetic, small-scale)
├── main.py                 <- Flask web application
├── Dockerfile
├── scripts/
│   ├── data.py             <- Data loading
│   ├── make_dataset.py     <- Data download
│   ├── synthetic_data.py   <- Synthetic data generation
│   ├── download_large_dataset.py  <- Bing image crawler (icrawler)
│   ├── clean_dataset.py    <- Dedup + validation + aspect filter
│   ├── train_large.py      <- Train on 4869 real images (MPS)
│   ├── eval_large.py       <- Re-evaluate + regenerate metrics.json
│   ├── update_demo_samples.py  <- Pick correct test-set samples for demo
│   ├── features.py         <- Interpretable feature extraction
│   ├── model.py            <- Three model implementations
│   ├── experiment.py       <- Experiment framework (robustness, gating, head-tail)
│   ├── eda.py              <- Exploratory data analysis
│   ├── regenerate_plots.py <- Regenerate all plots from metrics.json
│   └── make_pptx.py        <- Slide generator
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
- Paper: Yang et al., "A Large-Scale Car Dataset for Fine-Grained Categorization and Verification" [1]

For the final evaluation run, I crawled 5,006 real car photos from Bing Image Search using `icrawler` (5 classes × 5 keywords × 400 target images). After deduplication (MD5), corrupted-image filtering (PIL verify), and aspect-ratio filtering (0.5–3.0), 4,869 valid images remained. The split is 3,895 train / 974 test, with 5 demo samples selected from the test set (leaving 969 for evaluation). The repo also ships a synthetic-data generator for pipeline testing when real data isn't available.

## Evaluation Outputs

After running `python setup.py`, the following artifacts land in `data/outputs/`:

- `metrics.json` - all metrics (naive_majority, naive_random, classical, deep, robustness, head_tail, error_cases, meta)
- `model_comparison.png` - bar chart comparing three models across tasks
- `confusion_matrix.png` + `confusion_matrix.npy` - aggregate confusion matrix
- `robustness.png` - corruption robustness (gaussian_noise / motion_blur / jpeg_compression / pixelate)
- `confidence_curve.png` - confidence gating accuracy vs coverage
- `confidence_analysis.json` - per-threshold accuracy/coverage rows
- `sample_images.png` - sample images per class
- `plots/cm_*.png` - per-task per-model confusion matrices
- `plots/eda_dist_*.png` - class distribution bar charts

## Originality Statement

Original work for the Module 2 Project (Computer Vision course). What's new here:

1. **Multi-task learning + interpretable features together**: handcrafted features fused with deep features, which pushes accuracy up and stays interpretable - unlike single-task or pure-deep setups.
2. **Attribute correlation modeling**: the shared MobileNetV2 backbone + three heads pick up the correlations between type, doors, and seats.
3. **End-to-end interpretable system**: predicts attributes and writes a sentence explaining each one (e.g., "dominant color is blue", "aspect ratio 1.2 -> leans SUV").
4. **Experiments beyond accuracy**: the project covers corruption robustness, confidence gating, and head/tail class analysis - the things that actually matter for deployment.

I wrote all the code myself; only standard libraries/frameworks (PyTorch, scikit-learn, Flask, etc.) are used. CompCars is used under its non-commercial research license.

## Tech Stack

- **Deep Learning**: PyTorch, torchvision MobileNetV2
- **Classical ML**: scikit-learn, Random Forest
- **Feature Engineering**: scikit-image (HOG), LBP texture
- **Web Framework**: Flask, Gunicorn
- **Deployment**: Docker, HuggingFace Spaces
- **CI**: GitHub Actions (keep-alive)
