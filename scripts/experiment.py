"""实验框架：评估指标、多任务vs单任务对比、数据规模敏感性、错误分析."""
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
    """计算分类指标."""
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
    """绘制混淆矩阵."""
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


def evaluate_model(model, X, y, task_name, classes, model_name):
    """评估单个模型并保存结果."""
    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred, classes)
    # 混淆矩阵
    plot_confusion_matrix(
        y, y_pred, classes,
        f"{model_name} - {task_name} (Acc={metrics['accuracy']:.3f})",
        PLOTS_DIR / f"cm_{model_name}_{task_name}.png",
    )
    return metrics


def error_analysis(model, X, y, df_meta, task_name, classes, model_name, top_k=5):
    """错误分析：找出5个具体误预测."""
    y_pred = model.predict(X)
    errors = []
    for i, (true, pred) in enumerate(zip(y, y_pred)):
        if true != pred:
            row = df_meta.iloc[i] if hasattr(df_meta, "iloc") else {}
            errors.append({
                "index": int(i),
                "img_path": str(row.get("img_path", "")) if hasattr(row, "get") else "",
                "true_label": str(true),
                "pred_label": str(pred),
                "model_id": str(row.get("model_id", "")) if hasattr(row, "get") else "",
            })
    # 取前5个
    return errors[:top_k]


def data_size_sensitivity(model_factory, X_pool, y_pool, fractions=(0.1, 0.25, 0.5, 1.0)):
    """训练数据规模敏感性分析."""
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
    """多任务 vs 单任务性能对比."""
    results = {}
    for task in task_names:
        preds = deep_model.predict(X)
        if task in preds:
            metrics = compute_metrics(y_dict[task], preds[task])
            results[task] = metrics
    return results


def run_full_evaluation(results_dict, save=True):
    """汇总所有评估结果."""
    summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "models": results_dict,
    }
    if save:
        path = OUTPUTS_DIR / "metrics.json"
        path.write_text(json.dumps(summary, indent=2, default=str))
        print(f"评估结果已保存: {path}")
    return summary


if __name__ == "__main__":
    # 测试
    y_true = np.array(["sedan", "suv", "sedan", "suv", "sedan"])
    y_pred = np.array(["sedan", "sedan", "sedan", "suv", "suv"])
    print(compute_metrics(y_true, y_pred))
