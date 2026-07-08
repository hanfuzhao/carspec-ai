"""Exploratory data analysis: class distribution, sample visualization, basic statistics."""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.data import (
    get_splits, CAR_TYPES, DOOR_COUNTS, SEAT_COUNTS,
    TYPE2ID, DOOR2ID, SEAT2ID,
)

OUTPUTS_DIR = Path("data/outputs")
PLOTS_DIR = OUTPUTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df, task, classes, save_path):
    """Plot bar chart of class distribution for a task."""
    counts = df[task].value_counts().reindex(classes, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color="steelblue", edgecolor="black")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(counts.values) * 0.01,
                str(int(v)), ha="center", va="bottom", fontsize=10)
    ax.set_title(f"Class Distribution - {task}")
    ax.set_xlabel(task)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sample_images(df, n_per_class=3, save_path=None):
    from scripts.data import load_image, IMG_SIZE
    n_classes = len(CAR_TYPES)
    fig, axes = plt.subplots(n_per_class, n_classes, figsize=(n_classes * 2.5, n_per_class * 2.5))
    for col, cls in enumerate(CAR_TYPES):
        sub = df[df["car_type"] == cls].head(n_per_class)
        for row in range(n_per_class):
            ax = axes[row, col] if n_per_class > 1 else axes[col]
            if row < len(sub):
                try:
                    img = load_image(sub.iloc[row]["img_path"], IMG_SIZE)
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title(cls if row == 0 else "")
            ax.axis("off")
    plt.suptitle("Sample Images per Car Type")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_basic_stats(df):
    stats = {
        "n_total": int(len(df)),
        "n_models": int(df["model_id"].nunique()) if "model_id" in df.columns else 0,
        "tasks": {},
    }
    for task, classes in [("car_type", CAR_TYPES), ("door_count", DOOR_COUNTS), ("seat_count", SEAT_COUNTS)]:
        counts = df[task].value_counts().reindex(classes, fill_value=0)
        stats["tasks"][task] = {
            "classes": classes,
            "counts": counts.to_dict(),
            "majority_class": str(counts.idxmax()),
            "majority_ratio": float(counts.max() / len(df)),
            "n_classes": len(classes),
        }
    return stats


def run_eda():
    print("\n" + "=" * 60)
    print("EDA - Exploratory Data Analysis")
    print("=" * 60)
    train, val, test = get_splits()
    full = pd.concat([train, val, test], ignore_index=True)
    stats = compute_basic_stats(full)
    print(f"Total samples: {stats['n_total']:,}")
    print(f"Unique models: {stats['n_models']}")
    for task, info in stats["tasks"].items():
        print(f"\n{task}: majority={info['majority_class']} ({info['majority_ratio']:.1%})")
    for task, classes in [("car_type", CAR_TYPES), ("door_count", DOOR_COUNTS), ("seat_count", SEAT_COUNTS)]:
        plot_class_distribution(full, task, classes, PLOTS_DIR / f"eda_dist_{task}.png")
    sample_path = OUTPUTS_DIR / "sample_images.png"
    try:
        plot_sample_images(full, n_per_class=2, save_path=sample_path)
        print(f"Sample images saved: {sample_path}")
    except Exception as e:
        print(f"Sample image plotting skipped: {e}")
    stats_path = OUTPUTS_DIR / "eda_stats.json"
    import json
    stats_path.write_text(json.dumps(stats, indent=2, default=str))
    print(f"EDA stats saved: {stats_path}")
    return stats


if __name__ == "__main__":
    run_eda()
