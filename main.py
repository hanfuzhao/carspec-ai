"""CarSpec AI — Vehicle Multi-Attribute Intelligent Recognition System.

Flask Web App: Upload vehicle image -> Predict car type/door count/seat count + Interpretable feature explanation.

Deployment: HuggingFace Spaces / Docker
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

# Global model cache
MODELS = {}

MODEL_REPO = "HanfuZhao781/carspec-models"


def download_models():
    """Download model files from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_REPO,
            repo_type="model",
            local_dir=str(MODELS_DIR),
            local_dir_use_symlinks=False,
        )
        print("Model download complete")
    except Exception as e:
        print(f"Model download failed: {e}")


def load_models():
    """Load all trained models."""
    print("Loading models...")
    # If model files do not exist, download from Hub
    if not (MODELS_DIR / "classical_car_type.pkl").exists():
        download_models()
    # Classical models
    for task in ["car_type", "door_count", "seat_count"]:
        model = load_trained_model("classical", task=task)
        if model is not None:
            MODELS[f"classical_{task}"] = model
            print(f"  classical_{task} loaded")
    # Deep model (requires torch, may not be available at deployment)
    deep_path = MODELS_DIR / "deep_multitask.pt"
    if deep_path.exists():
        try:
            deep_model = DeepMultiTaskModel()
            deep_model.load(str(deep_path))
            MODELS["deep"] = deep_model
            print("  deep_multitask loaded")
        except Exception as e:
            print(f"  deep model load failed: {e}")
    # Naive baseline
    for task in ["car_type", "door_count", "seat_count"]:
        model = load_trained_model("naive", task=task)
        if model is not None:
            MODELS[f"naive_{task}"] = model
    print(f"Loaded {len(MODELS)} models")


def preprocess_image(file_storage, size=IMG_SIZE):
    """Preprocess uploaded image."""
    img = Image.open(file_storage).convert("RGB")
    img_resized = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr, img


def predict_with_classical(features):
    """Predict using Classical model."""
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
    """Predict using Deep model."""
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
        print(f"Deep prediction failed: {e}")
        return None


@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        img_array, original_img = preprocess_image(file)
        features = extract_all_features(img_array)
        # Classical prediction
        classical_results = predict_with_classical(features)
        # Deep prediction
        deep_results = predict_with_deep(img_array)
        # Interpretable explanation
        explanations = feature_importance_explanation(features)
        # Return results
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
    """Health check."""
    return jsonify({"status": "ok", "models_loaded": len(MODELS)})


@app.route("/models")
def models_info():
    """Model information."""
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
