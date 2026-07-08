"""Evaluate retrained models on large real-car dataset and regenerate metrics.json.

Loads saved models (deep_multitask.pt + classical_*.pkl + naive_*.pkl),
re-evaluates on the same 80/20 stratified split (random_state=42),
and writes a full metrics.json with all rubric-required fields.
"""
import os
import sys
import json
import time
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


def load_demo_samples():
    images, labels = [], []
    for car in CAR_TYPES:
        path = SAMPLES_DIR / f"sample_{car}.jpg"
        if path.exists():
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
            labels.append(car)
    return np.array(images), np.array(labels)


def predict_classical(rf, scaler, X):
    Xs = scaler.transform(X)
    proba = rf.predict_proba(Xs)
    idx = proba.argmax(axis=1)
    preds = [rf.classes_[i] for i in idx]
    confs = proba[np.arange(len(X)), idx]
    return preds, confs, proba


def predict_deep(model, images, device, task_classes):
    import torch
    model.eval()
    all_preds = {t: [] for t in task_classes}
    all_confs = {t: [] for t in task_classes}
    all_proba = {t: [] for t in task_classes}
    test_bs = 64
    with torch.no_grad():
        for i in range(0, len(images), test_bs):
            batch = images[i:i+test_bs]
            Xb = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(device)
            out = model(Xb)
            for task in task_classes:
                proba = torch.softmax(out[task], dim=1).cpu().numpy()
                idx2 = proba.argmax(axis=1)
                classes = task_classes[task]
                preds = [classes[j] for j in idx2]
                confs = proba[np.arange(len(batch)), idx2]
                all_preds[task].extend(preds)
                all_confs[task].extend(confs.tolist())
                all_proba[task].extend(proba.tolist())
    return all_preds, all_confs, all_proba


def calc_acc(preds, labels):
    return float(np.mean(np.array(preds) == np.array(labels)))


def top5_acc(proba_list, classes, yte, k=5):
    top5 = []
    for p in proba_list:
        sorted_idx = np.argsort(-np.array(p))
        top5.append([classes[j] for j in sorted_idx[:min(k, len(classes))]])
    return float(np.mean([yte[i] in top5[i] for i in range(len(yte))]))


def main():
    print("=" * 60)
    print("Evaluating saved models on large dataset")
    print("=" * 60)

    images, y_type, y_door, y_seat, img_paths = load_dataset()
    print(f"Loaded {len(images)} images: {Counter(y_type)}")

    idx = np.arange(len(images))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_type)
    X_train, X_test = images[train_idx], images[test_idx]
    yt_train, yt_test = y_type[train_idx], y_type[test_idx]
    yd_train, yd_test = y_door[train_idx], y_door[test_idx]
    ys_train, ys_test = y_seat[train_idx], y_seat[test_idx]
    test_paths = [img_paths[i] for i in test_idx]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    import hashlib
    demo_hashes = set()
    for cls in CAR_TYPES:
        p = SAMPLES_DIR / f"sample_{cls}.jpg"
        if p.exists():
            demo_hashes.add(hashlib.md5(p.read_bytes()).hexdigest())
    if demo_hashes and (SAMPLES_DIR / "selection_log.json").exists():
        kept = []
        removed = 0
        for i, p in enumerate(test_paths):
            h = hashlib.md5(Path(p).read_bytes()).hexdigest()
            if h in demo_hashes:
                removed += 1
                continue
            kept.append(i)
        if removed > 0:
            print(f"Removing {removed} demo sample(s) from test set to avoid overlap")
            test_idx = test_idx[kept]
            X_test = X_test[kept]
            yt_test = yt_test[kept]
            yd_test = yd_test[kept]
            ys_test = ys_test[kept]
            print(f"Clean test set: {len(X_test)} images")

    print("\n[1/5] Extracting features...")
    t0 = time.time()
    F_train = np.array([extract_all_features(img) for img in X_train])
    F_test = np.array([extract_all_features(img) for img in X_test])
    print(f"  Features: {F_train.shape}, took {time.time()-t0:.1f}s")

    task_classes = {"car_type": CAR_TYPES, "door_count": DOOR_COUNTS, "seat_count": SEAT_COUNTS}

    print("\n[2/5] Evaluating classical RF...")
    classical_results = {}
    for task, ytr, yte in [("car_type", yt_train, yt_test),
                            ("door_count", yd_train, yd_test),
                            ("seat_count", ys_train, ys_test)]:
        with open(MODELS_DIR / f"classical_{task}.pkl", "rb") as f:
            obj = pickle.load(f)
        rf = obj["model"]
        scaler = obj["scaler"]
        preds, confs, proba = predict_classical(rf, scaler, F_test)
        acc = calc_acc(preds, yte)
        t5 = top5_acc(proba.tolist(), list(rf.classes_), yte)
        classical_results[task] = {
            "accuracy": acc, "top5_accuracy": t5,
            "mean_confidence": float(np.mean(confs)),
            "predictions": preds, "confidences": confs.tolist(),
            "true_labels": yte.tolist(), "classes": list(rf.classes_),
        }
        print(f"  {task}: acc={acc:.4f}  top5={t5:.4f}")

    print("\n[3/5] Evaluating naive baselines...")
    naive_results = {}
    for task, ytr, yte in [("car_type", yt_train, yt_test),
                            ("door_count", yd_train, yd_test),
                            ("seat_count", ys_train, ys_test)]:
        with open(MODELS_DIR / f"naive_{task}.pkl", "rb") as f:
            obj = pickle.load(f)
        dummy = obj["model"]
        preds = dummy.predict(np.zeros((len(yte), 1)))
        acc = calc_acc(preds, yte)
        naive_results[task] = {"accuracy": acc, "top5_accuracy": acc}
        print(f"  {task}: acc={acc:.4f}")

    print("\n[4/5] Evaluating deep MobileNetV2...")
    import torch
    from scripts.model import DeepMultiTaskModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    deep_model = DeepMultiTaskModel()
    deep_model.load(str(MODELS_DIR / "deep_multitask.pt"))
    deep_model.model.to(device)
    deep_preds, deep_confs, deep_proba = predict_deep(
        deep_model.model, X_test, device, task_classes)
    deep_results = {}
    for task, yte in [("car_type", yt_test), ("door_count", yd_test), ("seat_count", ys_test)]:
        preds = deep_preds[task]
        confs = deep_confs[task]
        proba = deep_proba[task]
        acc = calc_acc(preds, yte)
        t5 = top5_acc(proba, task_classes[task], yte)
        deep_results[task] = {
            "accuracy": acc, "top5_accuracy": t5,
            "mean_confidence": float(np.mean(confs)),
            "predictions": preds, "confidences": confs,
            "true_labels": yte.tolist(), "classes": task_classes[task],
        }
        print(f"  {task}: acc={acc:.4f}  top5={t5:.4f}")

    print("\n[5/5] Computing robustness, head-tail, confidence gating...")
    robustness = {}
    np.random.seed(42)
    for severity in [0.05, 0.1, 0.2]:
        noisy = np.clip(X_test + np.random.randn(*X_test.shape) * severity, 0, 1)
        _, _, noisy_proba = predict_deep(deep_model.model, noisy, device, task_classes)
        noisy_preds = [task_classes["car_type"][np.argmax(p)] for p in noisy_proba["car_type"]]
        acc = calc_acc(noisy_preds, yt_test)
        robustness[f"gaussian_noise_{severity}"] = acc
        print(f"  noise {severity}: acc={acc:.4f}")
    robustness["mean_corruption_accuracy"] = float(np.mean(
        [robustness[k] for k in robustness if k.startswith("gaussian")]))

    class_acc = {}
    for cls in CAR_TYPES:
        mask = yt_test == cls
        if mask.sum() > 0:
            preds_arr = np.array(deep_results["car_type"]["predictions"])
            class_acc[cls] = calc_acc(preds_arr[mask], yt_test[mask])
    sorted_acc = sorted(class_acc.values())
    head_tail = {
        "head_accuracy": sorted_acc[-1] if sorted_acc else 0,
        "tail_accuracy": sorted_acc[0] if sorted_acc else 0,
        "gap": (sorted_acc[-1] - sorted_acc[0]) if sorted_acc else 0,
        "per_class": class_acc,
    }
    print(f"  Head: {head_tail['head_accuracy']:.4f}, Tail: {head_tail['tail_accuracy']:.4f}")

    conf_gating = {}
    deep_confs_arr = np.array(deep_results["car_type"]["confidences"])
    deep_preds_arr = np.array(deep_results["car_type"]["predictions"])
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        mask = deep_confs_arr >= thr
        if mask.sum() > 0:
            conf_gating[str(thr)] = {
                "coverage": float(mask.mean()),
                "accuracy": calc_acc(deep_preds_arr[mask], yt_test[mask]),
            }
        else:
            conf_gating[str(thr)] = {"coverage": 0.0, "accuracy": 0.0}

    error_cases = []
    for i in range(len(yt_test)):
        true_t = yt_test[i]
        pred_t = deep_results["car_type"]["predictions"][i]
        if true_t != pred_t:
            error_cases.append({
                "test_index": int(test_idx[i]),
                "true_label": true_t,
                "predicted": pred_t,
                "confidence": round(deep_results["car_type"]["confidences"][i], 4),
                "task": "car_type",
            })
    error_cases = error_cases[:15]

    print("\n[Demo] Testing on 5 static samples...")
    demo_imgs, demo_labels = load_demo_samples()
    demo_feats = np.array([extract_all_features(img) for img in demo_imgs])
    demo_classical = {}
    for task in ["car_type", "door_count", "seat_count"]:
        with open(MODELS_DIR / f"classical_{task}.pkl", "rb") as f:
            obj = pickle.load(f)
        preds, confs, _ = predict_classical(obj["model"], obj["scaler"], demo_feats)
        demo_classical[task] = {"preds": preds, "confs": confs.tolist()}
    demo_deep_preds, demo_deep_confs, _ = predict_deep(
        deep_model.model, demo_imgs, device, task_classes)
    demo_deep = {t: {"preds": demo_deep_preds[t], "confs": demo_deep_confs[t]}
                 for t in task_classes}
    for car, i in zip(demo_labels, range(len(demo_labels))):
        c_pred = demo_classical["car_type"]["preds"][i]
        d_pred = demo_deep["car_type"]["preds"][i]
        d_conf = demo_deep["car_type"]["confs"][i]
        ok = "OK" if d_pred == car else "XX"
        print(f"  {car:12s}  classical={c_pred:10s}  deep={d_pred:10s} ({d_conf:.2f}) {ok}")

    metrics = {
        "naive_majority": {t: {"accuracy": naive_results[t]["accuracy"],
                                "top5_accuracy": naive_results[t]["top5_accuracy"]}
                           for t in ["car_type", "door_count", "seat_count"]},
        "naive_random": {t: {"accuracy": 1.0 / len(task_classes[t]),
                              "top5_accuracy": 1.0}
                          for t in ["car_type", "door_count", "seat_count"]},
        "classical": {t: {"accuracy": classical_results[t]["accuracy"],
                          "top5_accuracy": classical_results[t]["top5_accuracy"],
                          "mean_confidence": classical_results[t]["mean_confidence"]}
                      for t in ["car_type", "door_count", "seat_count"]},
        "deep": {t: {"accuracy": deep_results[t]["accuracy"],
                     "top5_accuracy": deep_results[t]["top5_accuracy"],
                     "mean_confidence": deep_results[t]["mean_confidence"]}
                 for t in ["car_type", "door_count", "seat_count"]},
        "robustness": robustness,
        "head_tail": head_tail,
        "error_cases": error_cases,
        "meta": {
            "dataset": "Real car photos (Bing Image Search via icrawler) - 4869 images, 5 classes",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "img_size": IMG_SIZE,
            "feature_dim": F_train.shape[1],
            "deep_backbone": "MobileNetV2",
            "deep_epochs": 50,
            "eval_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "demo_samples": {
                "true_labels": demo_labels.tolist(),
                "classical_predictions": demo_classical["car_type"]["preds"],
                "deep_predictions": demo_deep["car_type"]["preds"],
                "deep_confidences": demo_deep["car_type"]["confs"],
            },
        },
        "confidence_gating": conf_gating,
        "data_size_sensitivity": {
            "comparison": {
                "small_dataset": {"n_images": 100, "deep_car_type_accuracy": 0.55, "deep_door_count_accuracy": 0.65, "deep_seat_count_accuracy": 0.65},
                "large_dataset": {"n_images": len(images), "deep_car_type_accuracy": deep_results["car_type"]["accuracy"], "deep_door_count_accuracy": deep_results["door_count"]["accuracy"], "deep_seat_count_accuracy": deep_results["seat_count"]["accuracy"]},
            },
            "note": "Scaling from 100 to 4869 real images improved car_type accuracy from 0.55 to {:.2f} (+{:.0%}).".format(deep_results["car_type"]["accuracy"], deep_results["car_type"]["accuracy"] - 0.55),
        },
    }

    out_path = OUTPUTS_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics written to {out_path}")
    print("\nSummary:")
    print(f"  Classical car_type: {classical_results['car_type']['accuracy']:.4f}")
    print(f"  Deep car_type:      {deep_results['car_type']['accuracy']:.4f}")
    print(f"  Deep door_count:    {deep_results['door_count']['accuracy']:.4f}")
    print(f"  Deep seat_count:    {deep_results['seat_count']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
