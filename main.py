"""CarSpec AI — 车辆多属性智能识别系统.

Flask Web 应用：上传车辆图片 → 预测车型类型/门数/座位数 + 可解释特征说明.

部署：HuggingFace Spaces / Docker
"""
import os
import io
import base64
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, render_template

from scripts.data import IMG_SIZE, CAR_TYPES, DOOR_COUNTS, SEAT_COUNTS, TYPE2ID, DOOR2ID, SEAT2ID
from scripts.features import extract_all_features, feature_importance_explanation, FEATURE_NAMES
from scripts.model import load_trained_model, DeepMultiTaskModel, MODELS_DIR

app = Flask(__name__, static_folder="static", template_folder="templates")

# 全局模型缓存
MODELS = {}


def load_models():
    """加载所有已训练模型."""
    print("加载模型...")
    # Classical 模型
    for task in ["car_type", "door_count", "seat_count"]:
        model = load_trained_model("classical", task=task)
        if model is not None:
            MODELS[f"classical_{task}"] = model
            print(f"  classical_{task} 已加载")
    # Deep 模型
    deep_path = MODELS_DIR / "deep_multitask.pt"
    if deep_path.exists():
        try:
            deep_model = DeepMultiTaskModel()
            deep_model.load(str(deep_path))
            MODELS["deep"] = deep_model
            print("  deep_multitask 已加载")
        except Exception as e:
            print(f"  deep 模型加载失败: {e}")
    # Naive 基线
    for task in ["car_type", "door_count", "seat_count"]:
        model = load_trained_model("naive", task=task)
        if model is not None:
            MODELS[f"naive_{task}"] = model
    print(f"已加载 {len(MODELS)} 个模型")


def preprocess_image(file_storage, size=IMG_SIZE):
    """预处理上传的图片."""
    img = Image.open(file_storage).convert("RGB")
    img_resized = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr, img


def predict_with_classical(features):
    """使用 Classical 模型预测."""
    results = {}
    for task, classes in [("car_type", CAR_TYPES), ("door_count", DOOR_COUNTS), ("seat_count", SEAT_COUNTS)]:
        key = f"classical_{task}"
        if key in MODELS:
            model = MODELS[key]
            proba = model.predict_proba(features.reshape(1, -1))[0]
            pred_idx = int(np.argmax(proba))
            results[task] = {
                "prediction": classes[pred_idx],
                "confidence": float(proba[pred_idx]),
                "probabilities": {classes[i]: float(p) for i, p in enumerate(proba)},
            }
    return results


def predict_with_deep(img_array):
    """使用 Deep 模型预测."""
    if "deep" not in MODELS:
        return None
    model = MODELS["deep"]
    X = np.expand_dims(img_array, axis=0)
    try:
        preds = model.predict_proba(X)
        results = {}
        task_classes = {"car_type": CAR_TYPES, "door_count": DOOR_COUNTS, "seat_count": SEAT_COUNTS}
        for task, classes in task_classes.items():
            if task in preds:
                proba = preds[task][0]
                pred_idx = int(np.argmax(proba))
                results[task] = {
                    "prediction": classes[pred_idx],
                    "confidence": float(proba[pred_idx]),
                    "probabilities": {classes[i]: float(p) for i, p in enumerate(proba)},
                }
        return results
    except Exception as e:
        print(f"Deep 预测失败: {e}")
        return None


@app.route("/")
def index():
    """主页."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """预测接口."""
    if "image" not in request.files:
        return jsonify({"error": "未上传图片"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    try:
        img_array, original_img = preprocess_image(file)
        features = extract_all_features(img_array)
        # Classical 预测
        classical_results = predict_with_classical(features)
        # Deep 预测
        deep_results = predict_with_deep(img_array)
        # 可解释说明
        explanations = feature_importance_explanation(features)
        # 返回结果
        response = {
            "success": True,
            "classical": classical_results,
            "deep": deep_results,
            "explanations": explanations,
            "features": dict(zip(FEATURE_NAMES, features.tolist())),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """健康检查."""
    return jsonify({"status": "ok", "models_loaded": len(MODELS)})


@app.route("/models")
def models_info():
    """模型信息."""
    return jsonify({
        "loaded": list(MODELS.keys()),
        "tasks": {
            "car_type": CAR_TYPES,
            "door_count": DOOR_COUNTS,
            "seat_count": SEAT_COUNTS,
        },
    })


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
