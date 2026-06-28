"""Experiment framework: metrics, robustness, confidence gating, head/tail analysis, error analysis."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUTS_DIR = Path("data/outputs")
PLOTS_DIR = OUTPUTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred, classes=None):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "n_samples": int(len(y_true)),
    }


def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return cm


def evaluate_model(model, X, y, task_name, classes, model_name):
    """Evaluate a single model and save confusion matrix + metrics."""
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred, classes)
    cm = plot_confusion_matrix(
        y, y_pred, classes,
        f"{model_name} - {task_name} (Acc={metrics['accuracy']:.3f})",
        PLOTS_DIR / f"cm_{model_name}_{task_name}.png",
    )
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def error_analysis(model, X, y, df_meta, task_name, classes, model_name, top_k=10):
    """Find misprediction cases with confidence and test_index.

    Required fields per case: true, predicted, confidence, test_index.
    """
    y_pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None
    errors = []
    for i, (true, pred) in enumerate(zip(y, y_pred)):
        if true != pred:
            conf = 0.0
            if proba is not None and i < len(proba):
                pred_idx = list(classes).index(pred) if pred in classes else 0
                conf = float(proba[i][pred_idx]) if pred_idx < proba.shape[1] else 0.0
            row_meta = {}
            if hasattr(df_meta, "iloc") and i < len(df_meta):
                row = df_meta.iloc[i]
                row_meta = row.to_dict() if hasattr(row, "to_dict") else {}
            errors.append({
                "test_index": int(i),
                "true": str(true),
                "predicted": str(pred),
                "confidence": conf,
                "task": task_name,
                "model": model_name,
                "img_path": str(row_meta.get("img_path", "")),
                "model_id": str(row_meta.get("model_id", "")),
            })
    return errors[:top_k]


def data_size_sensitivity(model_factory, X_pool, y_pool, fractions=(0.1, 0.25, 0.5, 1.0)):
    """Training data size sensitivity analysis."""
    results = []
    n = len(y_pool)
    for frac in fractions:
        n_samples = int(n * frac)
        idx = np.random.RandomState(42).choice(n, n_samples, replace=False)
        X_sub = X_pool[idx] if hasattr(X_pool, "__getitem__") else X_pool.iloc[idx]
        y_sub = y_pool[idx] if hasattr(y_pool, "__getitem__") else y_pool.iloc[idx]
        model = model_factory()
        model.fit(X_sub, y_sub)
        y_pred = model.predict(X_sub)
        metrics = compute_metrics(y_sub, y_pred)
        metrics["fraction"] = float(frac)
        metrics["n_train"] = n_samples
        results.append(metrics)
    return results


def multitask_vs_singletask(deep_model, X, y_dict, task_names):
    """Multi-task vs single-task performance comparison."""
    results = {}
    for task in task_names:
        preds = deep_model.predict(X)
        if task in preds:
            metrics = compute_metrics(y_dict[task], preds[task])
            results[task] = metrics
    return results


# ==================== Robustness Experiment ====================

CORRUPTIONS = ["gaussian_noise", "motion_blur", "jpeg_compression", "pixelate"]


def apply_gaussian_noise(img, sigma=0.1):
    """Add Gaussian noise."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1)


def apply_motion_blur(img, kernel_size=15):
    """Apply horizontal motion blur."""
    try:
        from scipy.ndimage import convolve
    except ImportError:
        k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        k[kernel_size // 2, :] = 1.0 / kernel_size
        h, w, c = img.shape
        out = np.zeros_like(img)
        for ch in range(c):
            out[..., ch] = _convolve2d(img[..., ch], k)
        return np.clip(out, 0, 1)
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size // 2, :] = 1.0 / kernel_size
    h, w, c = img.shape
    out = np.zeros_like(img)
    for ch in range(c):
        out[..., ch] = convolve(img[..., ch], k, mode="reflect")
    return np.clip(out, 0, 1)


def _convolve2d(arr, kernel):
    """Simple 2D convolution fallback."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            out += padded[i:i + arr.shape[0], j:j + arr.shape[1]] * kernel[i, j]
    return out


def apply_jpeg_compression(img, quality=20):
    """Apply JPEG compression artifact simulation."""
    from PIL import Image
    import io
    arr = (img * 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.array(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return out


def apply_pixelate(img, blocks=8):
    """Pixelate image by averaging blocks."""
    h, w, c = img.shape
    bh, bw = h // blocks, w // blocks
    out = img.copy()
    for i in range(0, h, bh):
        for j in range(0, w, bw):
            block = out[i:i + bh, j:j + bw]
            out[i:i + bh, j:j + bw] = block.mean(axis=(0, 1))
    return out


def apply_corruption(img, corruption_type, severity=1):
    """Apply a corruption to an image."""
    if corruption_type == "gaussian_noise":
        return apply_gaussian_noise(img, sigma=0.05 * severity)
    if corruption_type == "motion_blur":
        return apply_motion_blur(img, kernel_size=7 + 4 * severity)
    if corruption_type == "jpeg_compression":
        return apply_jpeg_compression(img, quality=max(10, 80 - 20 * severity))
    if corruption_type == "pixelate":
        return apply_pixelate(img, blocks=max(2, 16 - 4 * severity))
    return img


def robustness_experiment(model, X_test_images, y_test, classes, task="car_type",
                          severities=(1, 2, 3), save_plot=True):
    """Run corruption robustness experiment.

    Args:
        model: must support predict_proba for confidence
        X_test_images: N x H x W x 3 array of test images (for deep model)
                       or N x D feature matrix (for classical model)
        y_test: ground truth labels
        classes: list of class names
        task: task name
        severities: corruption severity levels

    Returns:
        dict: {corruption: {severity: accuracy}}, mean_corruption_accuracy
    """
    from scripts.features import extract_all_features
    is_image_input = len(X_test_images.shape) == 4 if hasattr(X_test_images, "shape") else False
    results = {}
    acc_per_corruption = {}
    for corruption in CORRUPTIONS:
        results[corruption] = {}
        accs = []
        for sev in severities:
            if is_image_input:
                corrupted = np.array([apply_corruption(img, corruption, sev) for img in X_test_images])
                try:
                    proba = model.predict_proba(corrupted)
                    y_pred = np.array([classes[int(np.argmax(p))] for p in proba])
                except Exception:
                    feats = np.array([extract_all_features(img) for img in corrupted])
                    y_pred = model.predict(feats)
            else:
                y_pred = model.predict(X_test_images)
            acc = float(accuracy_score(y_test, y_pred))
            results[corruption][f"severity_{sev}"] = acc
            accs.append(acc)
        acc_per_corruption[corruption] = float(np.mean(accs))
    mean_acc = float(np.mean(list(acc_per_corruption.values())))
    results["mean_corruption_accuracy"] = mean_acc
    results["per_corruption_mean"] = acc_per_corruption

    if save_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(severities))
        width = 0.2
        for i, corruption in enumerate(CORRUPTIONS):
            accs = [results[corruption][f"severity_{s}"] for s in severities]
            ax.bar(x + i * width, accs, width, label=corruption)
        ax.set_xlabel("Severity")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Robustness — {task}")
        ax.set_xticks(x + width * (len(CORRUPTIONS) - 1) / 2)
        ax.set_xticklabels([f"sev {s}" for s in severities])
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / "robustness.png", dpi=150, bbox_inches="tight")
        plt.close()
    return results


# ==================== Confidence Gating / Selective Prediction ====================

def confidence_gating_experiment(y_true, y_pred, confidences, thresholds=None):
    """Confidence gating: accuracy vs coverage trade-off.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        confidences: max softmax probability per sample
        thresholds: confidence thresholds to evaluate

    Returns:
        list of {threshold, accuracy, coverage} dicts
    """
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidences = np.asarray(confidences)
    rows = []
    for thr in thresholds:
        mask = confidences >= thr
        coverage = float(mask.mean())
        if mask.sum() > 0:
            acc = float(accuracy_score(y_true[mask], y_pred[mask]))
        else:
            acc = 0.0
        rows.append({
            "threshold": float(thr),
            "accuracy": acc,
            "coverage": coverage,
            "n_selected": int(mask.sum()),
        })

    fig, ax1 = plt.subplots(figsize=(9, 6))
    thr_arr = np.array([r["threshold"] for r in rows])
    acc_arr = np.array([r["accuracy"] for r in rows])
    cov_arr = np.array([r["coverage"] for r in rows])
    ax1.plot(thr_arr, acc_arr, "o-", color="steelblue", label="Accuracy")
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Accuracy", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(thr_arr, cov_arr, "s--", color="darkorange", label="Coverage")
    ax2.set_ylabel("Coverage", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 1.05)
    plt.title("Confidence Gating — Accuracy vs Coverage")
    fig.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confidence_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    with open(OUTPUTS_DIR / "confidence_analysis.json", "w") as f:
        json.dump(rows, f, indent=2)
    return rows


# ==================== Head/Tail Analysis ====================

def head_tail_analysis(y_true, y_pred, classes, top_k_ratio=0.5):
    """Head/tail (frequency-aware) analysis.

    Head classes = top-k most frequent in y_true.
    Tail classes = remaining less frequent.

    Returns dict with head_accuracy, tail_accuracy, gap.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    class_counts = pd.Series(y_true).value_counts().to_dict()
    sorted_classes = sorted(classes, key=lambda c: -class_counts.get(c, 0))
    n_head = max(1, int(len(sorted_classes) * top_k_ratio))
    head_classes = set(sorted_classes[:n_head])
    tail_classes = set(sorted_classes[n_head:])
    head_mask = np.array([yt in head_classes for yt in y_true])
    tail_mask = np.array([yt in tail_classes for yt in y_true])
    head_acc = float(accuracy_score(y_true[head_mask], y_pred[head_mask])) if head_mask.sum() > 0 else 0.0
    tail_acc = float(accuracy_score(y_true[tail_mask], y_pred[tail_mask])) if tail_mask.sum() > 0 else 0.0
    return {
        "head_classes": sorted(list(head_classes)),
        "tail_classes": sorted(list(tail_classes)),
        "head_accuracy": head_acc,
        "tail_accuracy": tail_acc,
        "gap": float(head_acc - tail_acc),
        "head_n": int(head_mask.sum()),
        "tail_n": int(tail_mask.sum()),
    }


# ==================== Aggregation ====================

def plot_model_comparison(results_dict, save_path):
    """Plot bar chart comparing models across tasks."""
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(results_dict.keys())
    tasks = ["car_type", "door_count", "seat_count"]
    x = np.arange(len(tasks))
    width = 0.8 / max(len(models), 1)
    for i, model_name in enumerate(models):
        accs = []
        for task in tasks:
            entry = results_dict[model_name]
            if isinstance(entry, dict) and task in entry:
                accs.append(entry[task].get("accuracy", 0.0))
            else:
                accs.append(0.0)
        ax.bar(x + i * width, accs, width, label=model_name)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Comparison across Tasks")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_full_evaluation(results_dict, save=True):
    """Aggregate all evaluation results into metrics.json."""
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        **results_dict,
    }
    if save:
        path = OUTPUTS_DIR / "metrics.json"
        path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"Evaluation results saved: {path}")
    return summary


if __name__ == "__main__":
    y_true = np.array(["sedan", "suv", "sedan", "suv", "sedan"])
    y_pred = np.array(["sedan", "sedan", "sedan", "suv", "suv"])
    print(compute_metrics(y_true, y_pred))
    confs = np.array([0.9, 0.4, 0.8, 0.7, 0.3])
    print(confidence_gating_experiment(y_true, y_pred, confs))
    print(head_tail_analysis(y_true, y_pred, ["sedan", "suv"]))
