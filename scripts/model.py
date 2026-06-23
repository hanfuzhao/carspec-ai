"""三个模型实现：Naive基线、Classical ML、Deep多任务学习.

统一接口：
- fit(X, y) 训练
- predict(X) 预测类别
- predict_proba(X) 预测概率
- save(path) / load(path) 持久化
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
from sklearn.pipeline import Pipeline

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
        return self.model.predict(np.zeros((len(X), 1)))

    def predict_proba(self, X):
        return self.model.predict_proba(np.zeros((len(X), 1)))

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
        # LinearSVC + 概率校准
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
        """返回特征重要性（仅RF支持）."""
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
# 3. Deep Learning 多任务模型
# ============================================================
class DeepMultiTaskModel:
    """ResNet50迁移学习 + 多任务分类头.

    共享 backbone，三个分类头分别预测：
    - car_type (5类)
    - door_count (3类)
    - seat_count (3类)
    """

    def __init__(self, backbone="resnet50", use_aux_features=False, aux_dim=50):
        self.backbone_name = backbone
        self.use_aux_features = use_aux_features
        self.aux_dim = aux_dim
        self.model = None
        self.classes_ = {
            "car_type": ["sedan", "suv", "mpv", "coupe", "hatchback"],
            "door_count": ["2", "4", "5"],
            "seat_count": ["2", "5", "7"],
        }

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import layers, Model, Input

        # 输入
        img_input = Input(shape=(224, 224, 3), name="image")
        inputs = [img_input]
        # ResNet50 backbone
        if self.backbone_name == "resnet50":
            base = tf.keras.applications.ResNet50(
                include_top=False, weights="imagenet", input_tensor=img_input
            )
        else:
            base = tf.keras.applications.MobileNetV2(
                include_top=False, weights="imagenet", input_tensor=img_input, alpha=0.75
            )
        base.trainable = False  # 先冻结
        x = layers.GlobalAveragePooling2D()(base.output)
        # 可选：拼接辅助特征
        if self.use_aux_features:
            aux_input = Input(shape=(self.aux_dim,), name="aux_features")
            inputs.append(aux_input)
            x = layers.Concatenate()([x, aux_input])
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        # 三个分类头
        heads = {}
        for task, n_classes in [("car_type", 5), ("door_count", 3), ("seat_count", 3)]:
            heads[task] = layers.Dense(n_classes, activation="softmax", name=task)(x)
        self.model = Model(inputs=inputs, outputs=heads)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss={task: "sparse_categorical_crossentropy" for task in heads},
            loss_weights={task: 1.0 for task in heads},
            metrics={task: "accuracy" for task in heads},
        )

    def fit(self, train_gen, val_gen, epochs=20, steps_per_epoch=None, validation_steps=None):
        import tensorflow as tf
        if self.model is None:
            self._build_model()
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
            tf.keras.callbacks.ModelCheckpoint(
                str(MODELS_DIR / "deep_best.h5"), save_best_only=True, save_weights_only=True
            ),
        ]
        history = self.model.fit(
            train_gen, validation_data=val_gen, epochs=epochs,
            steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
            callbacks=callbacks, verbose=1,
        )
        return history

    def predict(self, X, aux_features=None):
        inputs = [X] if not self.use_aux_features else [X, aux_features]
        preds = self.model.predict(inputs, verbose=0)
        return {task: np.argmax(preds[task], axis=1) for task in preds}

    def predict_proba(self, X, aux_features=None):
        inputs = [X] if not self.use_aux_features else [X, aux_features]
        return self.model.predict(inputs, verbose=0)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(str(path))
        return self


# ============================================================
# 工厂函数
# ============================================================
def get_model(model_name: str, task: str = "car_type", **kwargs):
    """根据名称获取模型实例."""
    if model_name == "naive":
        return NaiveBaseline(task=task)
    if model_name == "classical":
        return ClassicalModel(task=task, **kwargs)
    if model_name == "deep":
        return DeepMultiTaskModel(**kwargs)
    raise ValueError(f"未知模型: {model_name}")


def load_trained_model(model_name: str, task: str = "car_type"):
    """加载已训练的模型."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{model_name}_{task}.pkl"
    if model_name == "deep":
        path = MODELS_DIR / "deep_multitask.h5"
        if not path.exists():
            return None
        model = DeepMultiTaskModel()
        return model.load(path)
    if not path.exists():
        return None
    model = get_model(model_name, task=task)
    return model.load(path)


if __name__ == "__main__":
    print("可用模型: naive, classical, deep")
    print("Naive 测试:")
    m = NaiveBaseline()
    m.fit(None, np.array(["sedan", "sedan", "suv"]))
    print(f"  classes: {m.classes_}")
    print(f"  predict: {m.predict(np.zeros((2, 1)))}")
