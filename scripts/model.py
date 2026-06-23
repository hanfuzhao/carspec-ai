"""三个模型实现：Naive基线、Classical ML、Deep多任务学习.

统一接口：
- fit(X, y) 训练
- predict(X) 预测类别
- predict_proba(X) 预测概率
- save(path) / load(path) 持久化

Deep 模型使用 PyTorch + torchvision ResNet50.
"""
import os
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path("models")
SEED = 42


# ============================================================
# 1. Naive 基线模型
# ============================================================
class NaiveBaseline:
    """多数类基线：始终预测训练集中最常见的类别."""

    def __init__(self, task="car_type"):
        self.task = task
        self.model = DummyClassifier(strategy="most_frequent")
        self.classes_ = None

    def fit(self, X, y):
        self.model.fit(np.zeros((len(y), 1)), y)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return self.model.predict(np.zeros((n, 1)))

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else 1
        return self.model.predict_proba(np.zeros((n, 1)))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "task": self.task, "classes_": self.classes_}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.task = data["task"]
        self.classes_ = data["classes_"]
        return self


# ============================================================
# 2. Classical ML 模型
# ============================================================
class ClassicalModel:
    """经典ML：可解释视觉特征 + 随机森林."""

    def __init__(self, task="car_type", model_type="rf"):
        self.task = task
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.classes_ = None

    def _build(self):
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=100, max_depth=20, random_state=SEED, n_jobs=-1, class_weight="balanced"
            )
        svc = LinearSVC(C=1.0, random_state=SEED, class_weight="balanced", max_iter=2000)
        return CalibratedClassifierCV(svc, cv=3)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._build()
        self.model.fit(X_scaled, y)
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))

    def feature_importance(self):
        if self.model_type != "rf" or not hasattr(self.model, "feature_importances_"):
            return None
        return self.model.feature_importances_

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model, "scaler": self.scaler,
                "task": self.task, "model_type": self.model_type, "classes_": self.classes_,
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.task = data["task"]
        self.model_type = data["model_type"]
        self.classes_ = data["classes_"]
        return self


# ============================================================
# 3. Deep Learning 多任务模型 (PyTorch)
# ============================================================
class DeepMultiTaskModel:
    """ResNet50迁移学习 + 多任务分类头 (PyTorch).

    共享 backbone，三个分类头分别预测：
    - car_type (5类)
    - door_count (3类)
    - seat_count (3类)
    """

    def __init__(self, backbone="resnet50", use_aux_features=False, aux_dim=50, device=None):
        self.backbone_name = backbone
        self.use_aux_features = use_aux_features
        self.aux_dim = aux_dim
        self.device = device or ("cuda" if _torch_available() and _torch_cuda() else "cpu")
        self.model = None
        self.classes_ = {
            "car_type": ["sedan", "suv", "mpv", "coupe", "hatchback"],
            "door_count": ["2", "4", "5"],
            "seat_count": ["2", "5", "7"],
        }

    def _build_model(self):
        import torch
        import torch.nn as nn
        from torchvision import models

        class MultiTaskNet(nn.Module):
            def __init__(self, backbone_name, use_aux, aux_dim):
                super().__init__()
                if backbone_name == "resnet50":
                    base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                    feat_dim = base.fc.in_features
                    base.fc = nn.Identity()
                else:
                    base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
                    feat_dim = base.classifier[1].in_features
                    base.classifier = nn.Identity()
                self.backbone = base
                self.use_aux = use_aux
                in_dim = feat_dim + (aux_dim if use_aux else 0)
                self.shared = nn.Sequential(
                    nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.5)
                )
                self.head_car_type = nn.Linear(256, 5)
                self.head_door = nn.Linear(256, 3)
                self.head_seat = nn.Linear(256, 3)

            def forward(self, x, aux=None):
                feat = self.backbone(x)
                if self.use_aux and aux is not None:
                    feat = torch.cat([feat, aux], dim=1)
                shared = self.shared(feat)
                return {
                    "car_type": self.head_car_type(shared),
                    "door_count": self.head_door(shared),
                    "seat_count": self.head_seat(shared),
                }

        self.model = MultiTaskNet(self.backbone_name, self.use_aux_features, self.aux_dim)
        self.model.to(self.device)

    def fit(self, train_gen, val_gen, epochs=20, steps_per_epoch=None, validation_steps=None):
        import torch
        import torch.nn as nn
        from torch.optim import Adam
        if self.model is None:
            self._build_model()
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        patience, patience_counter = 5, 0
        history = {"train_loss": [], "val_acc": []}
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for step in range(steps_per_epoch or 100):
                try:
                    X, y = next(train_gen)
                except StopIteration:
                    break
                X_t = torch.FloatTensor(X).permute(0, 3, 1, 2).to(self.device)
                y_car = torch.LongTensor(y["car_type"]).to(self.device)
                y_door = torch.LongTensor(y["door_count"]).to(self.device)
                y_seat = torch.LongTensor(y["seat_count"]).to(self.device)
                optimizer.zero_grad()
                out = self.model(X_t)
                loss = (criterion(out["car_type"], y_car) +
                        criterion(out["door_count"], y_door) +
                        criterion(out["seat_count"], y_seat))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(steps_per_epoch or 100, 1)
            history["train_loss"].append(avg_loss)
            # 验证
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for step in range(validation_steps or 20):
                    try:
                        X, y = next(val_gen)
                    except StopIteration:
                        break
                    X_t = torch.FloatTensor(X).permute(0, 3, 1, 2).to(self.device)
                    out = self.model(X_t)
                    pred = out["car_type"].argmax(dim=1).cpu().numpy()
                    correct += (pred == y["car_type"]).sum()
                    total += len(pred)
            val_acc = correct / max(total, 1)
            history["val_acc"].append(float(val_acc))
            print(f"  Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - val_acc: {val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self._save_weights(MODELS_DIR / "deep_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        return history

    def _save_weights(self, path):
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path))

    def predict(self, X, aux_features=None):
        import torch
        if self.model is None:
            self._build_model()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).permute(0, 3, 1, 2).to(self.device)
            out = self.model(X_t)
        return {task: out[task].argmax(dim=1).cpu().numpy() for task in out}

    def predict_proba(self, X, aux_features=None):
        import torch
        if self.model is None:
            self._build_model()
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).permute(0, 3, 1, 2).to(self.device)
            out = self.model(X_t)
        result = {}
        for task in out:
            result[task] = torch.softmax(out[task], dim=1).cpu().numpy()
        return result

    def save(self, path):
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict() if self.model else None,
            "backbone": self.backbone_name,
            "use_aux": self.use_aux_features,
            "aux_dim": self.aux_dim,
        }, str(path))

    def load(self, path):
        import torch
        path = Path(path)
        if not path.exists():
            return None
        checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
        self.backbone_name = checkpoint.get("backbone", "resnet50")
        self.use_aux_features = checkpoint.get("use_aux", False)
        self.aux_dim = checkpoint.get("aux_dim", 50)
        self._build_model()
        if checkpoint.get("state_dict"):
            self.model.load_state_dict(checkpoint["state_dict"])
        return self


def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


def _torch_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ============================================================
# 工厂函数
# ============================================================
def get_model(model_name: str, task: str = "car_type", **kwargs):
    if model_name == "naive":
        return NaiveBaseline(task=task)
    if model_name == "classical":
        return ClassicalModel(task=task, **kwargs)
    if model_name == "deep":
        return DeepMultiTaskModel(**kwargs)
    raise ValueError(f"未知模型: {model_name}")


def load_trained_model(model_name: str, task: str = "car_type"):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if model_name == "deep":
        path = MODELS_DIR / "deep_multitask.pt"
        if not path.exists():
            return None
        model = DeepMultiTaskModel()
        return model.load(path)
    path = MODELS_DIR / f"{model_name}_{task}.pkl"
    if not path.exists():
        return None
    model = get_model(model_name, task=task)
    return model.load(path)


if __name__ == "__main__":
    print("可用模型: naive, classical, deep")
    m = NaiveBaseline()
    m.fit(None, np.array(["sedan", "sedan", "suv"]))
    print(f"  classes: {m.classes_}")
    print(f"  predict: {m.predict(np.zeros((2, 1)))}")
