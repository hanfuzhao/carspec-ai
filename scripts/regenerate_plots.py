"""Regenerate all data/outputs/ plots and JSON files from the new metrics.json.

Reads metrics.json (re-evaluated on 4869-image dataset) and produces:
- model_comparison.png
- confusion_matrix.png / .npy
- robustness.png
- confidence_curve.png
- confidence_analysis.json
- sample_images.png
- eda_stats.json
"""
import os
import sys
import json
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.data import IMG_SIZE, CAR_TYPES, DOOR_COUNTS, SEAT_COUNTS, TYPE2ID, DOOR2ID, SEAT2ID
from scripts.features import extract_all_features

DATA_DIR = ROOT / "data" / "real_cars_large"
SAMPLES_DIR = ROOT / "static" / "samples"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "data" / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DOOR_MAP = {"sedan": "4", "suv": "5", "mpv": "5", "coupe": "2", "hatchback": "5"}
SEAT_MAP = {"sedan": "5", "suv": "5", "mpv": "7", "coupe": "2", "hatchback": "5"}


def load_dataset():
    images, y_type, y_door, y_seat = [], [], [], []
    img_paths = []
    for cls_dir in sorted(DATA_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        car_type = cls_dir.name
        if car_type not in CAR_TYPES:
            continue
        for img_path in sorted(cls_dir.glob("*.jpg")):
            try:
                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr)
                y_type.append(car_type)
                y_door.append(DOOR_MAP[car_type])
                y_seat.append(SEAT_MAP[car_type])
                img_paths.append(str(img_path))
            except Exception:
                continue
    return (np.array(images), np.array(y_type),
            np.array(y_door), np.array(y_seat), img_paths)


def plot_model_comparison(metrics):
    fig, ax = plt.subplots(figsize=(9, 5))
    models = ["Naive", "Classical", "Deep"]
    tasks = ["car_type", "door_count", "seat_count"]
    task_labels = ["Car Type", "Door Count", "Seat Count"]
    x = np.arange(len(tasks))
    width = 0.25
    colors = ["#9CA3AF", "#3B82F6", "#DC2626"]
    for i, m in enumerate(models):
        key = "naive_majority" if m == "Naive" else m.lower()
        accs = [metrics[key][t]["accuracy"] for t in tasks]
        bars = ax.bar(x + i * width, accs, width, label=m, color=colors[i])
        for b, v in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison Across Tasks")
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  model_comparison.png")


def plot_confusion_matrix(metrics):
    images, y_type, _, _, _ = load_dataset()
    idx = np.arange(len(images))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_type)
    yt_test = y_type[test_idx]
    X_test = images[test_idx]

    import torch
    import hashlib
    from scripts.model import DeepMultiTaskModel
    demo_hashes = set()
    for cls in CAR_TYPES:
        p = SAMPLES_DIR / f"sample_{cls}.jpg"
        if p.exists():
            demo_hashes.add(hashlib.md5(p.read_bytes()).hexdigest())
    if demo_hashes and (SAMPLES_DIR / "selection_log.json").exists():
        kept = []
        for i, p in enumerate([DATA_DIR / yt_test[j] / f"{str(test_idx[j]).zfill(6)}.jpg" for j in range(len(test_idx))]):
            if p.exists():
                h = hashlib.md5(p.read_bytes()).hexdigest()
                if h in demo_hashes:
                    continue
            kept.append(i)
        if kept:
            test_idx = test_idx[kept]
            yt_test = yt_test[kept]
            X_test = X_test[kept]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    deep_model = DeepMultiTaskModel()
    deep_model.load(str(MODELS_DIR / "deep_multitask.pt"))
    deep_model.model.to(device)
    deep_model.model.eval()

    all_preds = []
    test_bs = 64
    with torch.no_grad():
        for i in range(0, len(X_test), test_bs):
            batch = X_test[i:i+test_bs]
            Xb = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(device)
            out = deep_model.model(Xb)
            pred = out["car_type"].argmax(dim=1).cpu().numpy()
            all_preds.extend([CAR_TYPES[j] for j in pred])

    cm = sk_confusion_matrix(yt_test, all_preds, labels=CAR_TYPES)
    np.save(OUTPUTS_DIR / "confusion_matrix.npy", cm)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CAR_TYPES)))
    ax.set_yticks(range(len(CAR_TYPES)))
    ax.set_xticklabels(CAR_TYPES, rotation=45, ha="right")
    ax.set_yticklabels(CAR_TYPES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Deep Model (car_type)")
    thresh = cm.max() / 2
    for i in range(len(CAR_TYPES)):
        for j in range(len(CAR_TYPES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  confusion_matrix.png + .npy")


def plot_robustness(metrics):
    rob = metrics["robustness"]
    severities = [0.05, 0.1, 0.2]
    accs = [rob[f"gaussian_noise_{s}"] for s in severities]
    clean_acc = metrics["deep"]["car_type"]["accuracy"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = ["clean"] + [f"σ={s}" for s in severities]
    y = [clean_acc] + accs
    bars = ax.bar(x, y, color=["#10B981", "#3B82F6", "#F59E0B", "#DC2626"])
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.axhline(y=rob["mean_corruption_accuracy"], color="gray", linestyle="--",
               label=f"Mean under noise: {rob['mean_corruption_accuracy']:.3f}")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness Under Gaussian Noise (Deep Model, car_type)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "robustness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  robustness.png")


def plot_confidence_curve(metrics):
    gating = metrics["confidence_gating"]
    thresholds = sorted([float(t) for t in gating.keys()])
    coverages = [gating[str(t)]["coverage"] for t in thresholds]
    accuracies = [gating[str(t)]["accuracy"] for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, coverages, "o-", label="Coverage", color="#3B82F6", linewidth=2)
    ax.plot(thresholds, accuracies, "s-", label="Accuracy on committed", color="#DC2626", linewidth=2)
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Value")
    ax.set_title("Confidence Gating: Coverage vs Accuracy (Deep Model, car_type)")
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confidence_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    analysis = [
        {"threshold": t, "accuracy": gating[str(t)]["accuracy"],
         "coverage": gating[str(t)]["coverage"],
         "n_selected": int(gating[str(t)]["coverage"] * 969)}
        for t in thresholds
    ]
    with open(OUTPUTS_DIR / "confidence_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("  confidence_curve.png + confidence_analysis.json")


def plot_sample_images():
    images, y_type, _, _, _ = load_dataset()
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, cls in enumerate(CAR_TYPES):
        mask = y_type == cls
        idxs = np.where(mask)[0]
        for j in range(2):
            if j < len(idxs):
                ax = axes[j, i]
                ax.imshow(images[idxs[j]])
                ax.set_title(cls, fontsize=11)
                ax.axis("off")
    plt.suptitle("Sample Images per Class (2 each)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "sample_images.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  sample_images.png")


def write_eda_stats():
    images, y_type, y_door, y_seat, _ = load_dataset()
    counts = Counter(y_type)
    stats = {
        "n_total": len(images),
        "n_classes": len(CAR_TYPES),
        "tasks": {
            "car_type": {
                "classes": CAR_TYPES,
                "counts": {c: int(counts.get(c, 0)) for c in CAR_TYPES},
                "majority_class": counts.most_common(1)[0][0] if counts else None,
            },
            "door_count": {
                "classes": DOOR_COUNTS,
                "counts": {c: int(Counter(y_door).get(c, 0)) for c in DOOR_COUNTS},
            },
            "seat_count": {
                "classes": SEAT_COUNTS,
                "counts": {c: int(Counter(y_seat).get(c, 0)) for c in SEAT_COUNTS},
            },
        },
        "img_size": IMG_SIZE,
        "dataset_source": "Bing Image Search via icrawler (4869 real car photos)",
    }
    with open(OUTPUTS_DIR / "eda_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("  eda_stats.json")


def main():
    print("=" * 60)
    print("Regenerating all plots from new metrics.json")
    print("=" * 60)
    with open(OUTPUTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    print("\n[1/6] Model comparison plot...")
    plot_model_comparison(metrics)

    print("\n[2/6] Confusion matrix...")
    plot_confusion_matrix(metrics)

    print("\n[3/6] Robustness plot...")
    plot_robustness(metrics)

    print("\n[4/6] Confidence curve + analysis...")
    plot_confidence_curve(metrics)

    print("\n[5/6] Sample images...")
    plot_sample_images()

    print("\n[6/6] EDA stats...")
    write_eda_stats()

    print("\nAll plots regenerated:")
    for f in sorted(OUTPUTS_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
