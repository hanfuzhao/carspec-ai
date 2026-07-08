"""Train MobileNetV2 multi-task model on large real-car dataset with MPS acceleration.

Uses 10000+ images, 80/20 split, 50 epochs, MobileNetV2 backbone with last 4 blocks unfrozen.
Saves model to models/deep_multitask.pt and classical models to models/classical_*.pkl.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.data import IMG_SIZE, CAR_TYPES, DOOR_COUNTS, SEAT_COUNTS, TYPE2ID, DOOR2ID, SEAT2ID
from scripts.features import extract_all_features

DATA_DIR = ROOT / "data" / "real_cars_large"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DOOR_MAP = {"sedan": "4", "suv": "5", "mpv": "5", "coupe": "2", "hatchback": "5"}
SEAT_MAP = {"sedan": "5", "suv": "5", "mpv": "7", "coupe": "2", "hatchback": "5"}


def load_dataset():
    images, y_type, y_door, y_seat = [], [], [], []
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
            except Exception:
                continue
    return (np.array(images), np.array(y_type),
            np.array(y_door), np.array(y_seat))


def main():
    print("=" * 60)
    print("Training on large real-car dataset (MPS accelerated)")
    print("=" * 60)

    images, y_type, y_door, y_seat = load_dataset()
    n = len(images)
    print(f"Loaded {n} images: {Counter(y_type)}")

    if n < 100:
        print("ERROR: too few images, aborting")
        return

    idx = np.arange(n)
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_type)
    X_train, X_test = images[train_idx], images[test_idx]
    yt_train, yt_test = y_type[train_idx], y_type[test_idx]
    yd_train, yd_test = y_door[train_idx], y_door[test_idx]
    ys_train, ys_test = y_seat[train_idx], y_seat[test_idx]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("\n[1/4] Extracting handcrafted features...")
    t0 = time.time()
    F_train = np.array([extract_all_features(img) for img in X_train])
    F_test = np.array([extract_all_features(img) for img in X_test])
    print(f"  Features: {F_train.shape}, took {time.time()-t0:.1f}s")

    print("\n[2/4] Training classical RF (3 tasks)...")
    for task, ytr, yte in [("car_type", yt_train, yt_test),
                            ("door_count", yd_train, yd_test),
                            ("seat_count", ys_train, ys_test)]:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(F_train)
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42,
                                    n_jobs=-1, class_weight="balanced")
        rf.fit(Xs, ytr)
        preds = rf.predict(scaler.transform(F_test))
        acc = float(np.mean(preds == yte))
        print(f"  {task}: acc={acc:.3f}")
        with open(MODELS_DIR / f"classical_{task}.pkl", "wb") as f:
            pickle.dump({"model": rf, "scaler": scaler, "task": task,
                         "model_type": "rf", "classes_": list(rf.classes_)}, f)

    print("\n  Training naive baselines...")
    for task, ytr in [("car_type", yt_train), ("door_count", yd_train), ("seat_count", ys_train)]:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(ytr), 1)), ytr)
        with open(MODELS_DIR / f"naive_{task}.pkl", "wb") as f:
            pickle.dump({"model": dummy, "task": task, "classes_": list(dummy.classes_)}, f)

    print("\n[3/4] Training deep MobileNetV2 (MPS) - v2 improved...")
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torchvision import models, transforms

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            feat_dim = base.classifier[1].in_features
            base.classifier = nn.Identity()
            self.backbone = base
            self.shared = nn.Sequential(
                nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3)
            )
            self.head_car_type = nn.Linear(256, 5)
            self.head_door = nn.Linear(256, 3)
            self.head_seat = nn.Linear(256, 3)

        def forward(self, x):
            f = self.backbone(x)
            s = self.shared(f)
            return {"car_type": self.head_car_type(s),
                    "door_count": self.head_door(s),
                    "seat_count": self.head_seat(s)}

    model = Net().to(device)
    for name, p in model.backbone.named_parameters():
        p.requires_grad = any(b in name for b in
                              ["features.10", "features.11", "features.12", "features.13",
                               "features.14", "features.15", "features.16", "features.17",
                               "features.18", "features.19", "classifier"])

    yt = np.array([TYPE2ID[t] for t in yt_train])
    yd = np.array([DOOR2ID[d] for d in yd_train])
    ys = np.array([SEAT2ID[s] for s in ys_train])

    aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
    ])

    Xt = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    n_train = len(X_train)
    bs = 32
    epochs = 80
    patience = 15
    patience_counter = 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    print(f"  Training {epochs} epochs, batch size {bs}, patience {patience}")
    print(f"  Unfreezing features.10-19 (10 blocks), label smoothing 0.1, cosine lr")
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_train)
        total_loss = 0
        n_batches = 0
        for i in range(0, n_train, bs):
            idx = perm[i:i+bs]
            xb = torch.stack([aug(Xt[j]) for j in idx]).to(device)
            opt.zero_grad()
            out = model(xb)
            loss = (crit(out["car_type"], torch.LongTensor(yt[idx]).to(device)) +
                    crit(out["door_count"], torch.LongTensor(yd[idx]).to(device)) +
                    crit(out["seat_count"], torch.LongTensor(ys[idx]).to(device)))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            test_bs = 64
            for i in range(0, len(X_test), test_bs):
                batch = X_test[i:i+test_bs]
                Xb = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(device)
                out = model(Xb)
                pred = out["car_type"].argmax(dim=1).cpu().numpy()
                yt_batch = np.array([TYPE2ID[t] for t in yt_test[i:i+test_bs]])
                correct += (pred == yt_batch).sum()
                total += len(pred)
        acc = float(correct / max(total, 1))

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save({
                "state_dict": model.state_dict(),
                "backbone": "mobilenet",
                "use_aux": False,
                "aux_dim": 50,
            }, str(MODELS_DIR / "deep_multitask.pt"))
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - loss: {total_loss/n_batches:.4f} - val_acc: {acc:.4f} - best: {best_acc:.4f} - lr: {scheduler.get_last_lr()[0]:.6f}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience {patience})")
            break

    print(f"\n  Best val accuracy: {best_acc:.4f}")
    print(f"  Model saved: {MODELS_DIR / 'deep_multitask.pt'}")

    print("\n[4/4] Final evaluation on test set...")
    model.eval()
    all_preds = {}
    with torch.no_grad():
        test_bs = 64
        for task in ["car_type", "door_count", "seat_count"]:
            all_preds[task] = []
            for i in range(0, len(X_test), test_bs):
                batch = X_test[i:i+test_bs]
                Xb = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(device)
                out = model(Xb)
                proba = torch.softmax(out[task], dim=1).cpu().numpy()
                idx = proba.argmax(axis=1)
                classes = CAR_TYPES if task == "car_type" else (DOOR_COUNTS if task == "door_count" else SEAT_COUNTS)
                preds = [classes[j] for j in idx]
                all_preds[task].extend(preds)
    for task, yte in [("car_type", yt_test), ("door_count", yd_test), ("seat_count", ys_test)]:
        preds = all_preds[task]
        acc = float(np.mean(np.array(preds) == np.array(yte)))
        print(f"  {task}: acc={acc:.4f}")

    print("\nDone. Models saved to models/")


if __name__ == "__main__":
    main()
