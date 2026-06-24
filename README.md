---
title: CarSpec AI
emoji: 🚗
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# CarSpec AI — Vehicle Multi-Attribute Intelligent Recognition System

> Module 2 Project · Computer Vision · Based on the CompCars Dataset

## Project Introduction

CarSpec AI is a computer vision-based vehicle multi-attribute recognition system. Upload a vehicle exterior photo, and the system simultaneously predicts three attributes: **vehicle type**, **door count**, and **seat count**, and provides interpretable visual feature analysis.

## Core Innovations

1. **Multi-task Joint Learning**: MobileNetV2 shared backbone + three classification heads, leveraging task correlations to improve performance
2. **Interpretable Visual Features**: Extract 50+ handcrafted features (color histogram, HOG, texture, body proportions, symmetry), providing better interpretability than pure CNNs
3. **Three-Model Comparison**: Naive baseline / Classical ML (Random Forest) / Deep Learning (MobileNetV2 multi-task)

## Online Demo

Visit the HuggingFace Space: https://hanfuzhao781-carspec-ai.hf.space

## Project Structure

```
├── README.md
├── REPORT.md               <- Full project report
├── requirements.txt
├── Makefile
├── setup.py                <- Training pipeline
├── main.py                 <- Flask web application
├── Dockerfile
├── scripts/
│   ├── data.py             <- Data loading
│   ├── make_dataset.py     <- Data download
│   ├── synthetic_data.py   <- Synthetic data generation
│   ├── features.py         <- Interpretable feature extraction
│   ├── model.py            <- Three model implementations
│   └── experiment.py       <- Experiment framework
├── models/                 <- Trained models
├── data/
│   ├── raw/                <- Raw data
│   ├── processed/          <- Processed data
│   └── outputs/            <- Output results
├── static/                 <- Frontend assets
├── templates/              <- HTML templates
└── .github/                <- CI/PR templates
```

## Three Model Locations

| Model | Code Location | Trained Model |
|------|---------|-----------|
| Naive Baseline | `scripts/model.py` → `NaiveBaseline` class | `models/naive_*.pkl` |
| Classical ML | `scripts/model.py` → `ClassicalModel` class | `models/classical_*.pkl` |
| Deep Learning | `scripts/model.py` → `DeepMultiTaskModel` class | `models/deep_multitask.pt` |

## Dataset

**CompCars** (CVPR 2015): 136,726 whole-vehicle images, annotated with vehicle type, door count, seat count, maximum speed, displacement, and other attributes.

- Official website: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
- Paper: Yang et al., "A Large-Scale Car Dataset for Fine-Grained Categorization and Verification"

## Tech Stack

- **Deep Learning**: PyTorch, torchvision MobileNetV2
- **Classical ML**: scikit-learn, Random Forest
- **Feature Engineering**: scikit-image (HOG), LBP texture
- **Web Framework**: Flask
- **Deployment**: Docker, HuggingFace Spaces
