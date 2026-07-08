"""Replace static/samples/ with correctly-predicted images from test set.

For each car class, finds one test set image that the deep model predicts correctly
with high confidence, copies it to static/samples/sample_{class}.jpg.
"""
import os
import sys
import shutil
import time
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.data import IMG_SIZE, CAR_TYPES, DOOR_COUNTS, SEAT_COUNTS, TYPE2ID, DOOR2ID, SEAT2ID
from scripts.features import extract_all_features

DATA_DIR = ROOT / "data" / "real_cars_large"
SAMPLES_DIR = ROOT / "static" / "samples"
MODELS_DIR = ROOT / "models"

DOOR_MAP = {"sedan": "4", "suv": "5", "mpv": "5", "coupe": "2", "hatchback": "5"}
SEAT_MAP = {"sedan": "5", "suv": "5", "mpv": "7", "coupe": "2", "hatchback": "5"}


def main():
    import torch
    from scripts.model import DeepMultiTaskModel

    print("=" * 60)
    print("Selecting correct demo samples from test set")
    print("=" * 60)

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
                img_paths.append(img_path)
            except Exception:
                continue
    images = np.array(images)
    y_type = np.array(y_type)
    y_door = np.array(y_door)
    y_seat = np.array(y_seat)
    print(f"Loaded {len(images)} images")

    idx = np.arange(len(images))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_type)
    print(f"Test set: {len(test_idx)} images")

    X_test = images[test_idx]
    yt_test = y_type[test_idx]
    test_paths = [img_paths[i] for i in test_idx]

    print("\nLoading deep model...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    deep_model = DeepMultiTaskModel()
    deep_model.load(str(MODELS_DIR / "deep_multitask.pt"))
    deep_model.model.to(device)
    deep_model.model.eval()

    print("Predicting on test set...")
    all_preds = []
    all_confs = []
    test_bs = 64
    with torch.no_grad():
        for i in range(0, len(X_test), test_bs):
            batch = X_test[i:i+test_bs]
            Xb = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(device)
            out = deep_model.model(Xb)
            proba = torch.softmax(out["car_type"], dim=1).cpu().numpy()
            idx2 = proba.argmax(axis=1)
            preds = [CAR_TYPES[j] for j in idx2]
            confs = proba[np.arange(len(batch)), idx2]
            all_preds.extend(preds)
            all_confs.extend(confs.tolist())

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    backup_dir = SAMPLES_DIR.parent / "samples_backup"
    backup_dir.mkdir(exist_ok=True)
    for old in SAMPLES_DIR.glob("*.jpg"):
        shutil.copy(old, backup_dir / old.name)
    print(f"Backed up old samples to {backup_dir}")

    print("\nSelecting one high-confidence correct sample per class:")
    selected = {}
    for cls in CAR_TYPES:
        candidates = []
        for i in range(len(test_idx)):
            if yt_test[i] == cls and all_preds[i] == cls:
                candidates.append((i, all_confs[i]))
        candidates.sort(key=lambda x: -x[1])
        if candidates:
            top = candidates[:5]
            chosen = top[np.random.RandomState(42 + CAR_TYPES.index(cls)).randint(len(top))]
            i, conf = chosen
            src = test_paths[i]
            dst = SAMPLES_DIR / f"sample_{cls}.jpg"
            shutil.copy(src, dst)
            selected[cls] = {"path": str(src), "confidence": conf, "true": cls, "pred": all_preds[i]}
            print(f"  {cls:12s}  conf={conf:.4f}  src={src.name}  ->  {dst.name}")
        else:
            print(f"  {cls:12s}  NO correct predictions in test set!")

    print("\nVerifying selected samples by reloading model:")
    verify_imgs = []
    verify_labels = []
    for cls in CAR_TYPES:
        p = SAMPLES_DIR / f"sample_{cls}.jpg"
        if p.exists():
            img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            verify_imgs.append(arr)
            verify_labels.append(cls)
    verify_imgs = np.array(verify_imgs)
    with torch.no_grad():
        Xb = torch.FloatTensor(verify_imgs).permute(0, 3, 1, 2).to(device)
        out = deep_model.model(Xb)
        proba = torch.softmax(out["car_type"], dim=1).cpu().numpy()
    print(f"\nFinal verification on {len(verify_imgs)} samples:")
    all_ok = True
    for i, cls in enumerate(verify_labels):
        pred_idx = int(np.argmax(proba[i]))
        pred = CAR_TYPES[pred_idx]
        conf = float(proba[i][pred_idx])
        ok = "OK" if pred == cls else "XX"
        if pred != cls:
            all_ok = False
        print(f"  {cls:12s}  pred={pred:12s}  conf={conf:.4f}  {ok}")
    if all_ok:
        print("\nAll demo samples correctly predicted!")
    else:
        print("\nWARNING: some samples still misclassified")

    metrics_note = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selected_from": "test set (974 images, deep model correct + high confidence)",
        "samples": selected,
    }
    with open(SAMPLES_DIR / "selection_log.json", "w") as f:
        import json
        json.dump(metrics_note, f, indent=2, default=str)
    print(f"\nSelection log: {SAMPLES_DIR / 'selection_log.json'}")


if __name__ == "__main__":
    main()
