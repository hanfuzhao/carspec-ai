"""CarSpec AI - Vehicle Multi-Attribute Recognition.

Flask app: upload a vehicle photo, get back type/door count/seat count + feature explanation.

Deployed on HuggingFace Spaces / Docker.
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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODELS = {}

MODEL_REPO = "HanfuZhao781/carspec-models"

TASK_CLASSES = {
    "car_type": CAR_TYPES,
    "door_count": DOOR_COUNTS,
    "seat_count": SEAT_COUNTS,
}


def download_models():
    """Download model files from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
        import shutil
        if MODELS_DIR.exists():
            for f in MODELS_DIR.iterdir():
                if f.is_file() and f.suffix in (".pkl", ".pt"):
                    f.unlink()
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading models to {MODELS_DIR}...")
        snapshot_download(
            repo_id=MODEL_REPO,
            repo_type="model",
            local_dir=str(MODELS_DIR),
            force_download=True,
        )
        print("Model download complete")
        downloaded = list(MODELS_DIR.iterdir())
        print(f"Files in models dir: {[f.name for f in downloaded]}")
    except Exception as e:
        print(f"Model download failed: {e}")
        import traceback
        traceback.print_exc()


def load_models():
    print("Loading models...")
    required = ["classical_car_type.pkl", "deep_multitask.pt", "naive_car_type.pkl"]
    expected_sizes = {
        "classical_car_type.pkl": 9_000_000,
        "deep_multitask.pt": 10_000_000,
        "naive_car_type.pkl": 500,
    }
    need_download = False
    for f in required:
        p = MODELS_DIR / f
        if not p.exists():
            need_download = True
            break
        min_size = expected_sizes.get(f, 0)
        if p.stat().st_size < min_size:
            print(f"  {f} size {p.stat().st_size} < expected {min_size}, redownloading")
            need_download = True
            break
    if need_download:
        download_models()
    for task in ["car_type", "door_count", "seat_count"]:
        model = load_trained_model("classical", task=task)
        if model is not None:
            MODELS[f"classical_{task}"] = model
            print(f"  classical_{task} loaded")
    deep_path = MODELS_DIR / "deep_multitask.pt"
    if deep_path.exists():
        try:
            deep_model = DeepMultiTaskModel()
            deep_model.load(str(deep_path))
            MODELS["deep"] = deep_model
            print("  deep_multitask loaded")
        except Exception as e:
            print(f"  deep model load failed: {e}")
    for task in ["car_type", "door_count", "seat_count"]:
        model = load_trained_model("naive", task=task)
        if model is not None:
            MODELS[f"naive_{task}"] = model
    print(f"Loaded {len(MODELS)} models")


def preprocess_image(file_storage, size=IMG_SIZE):
    img = Image.open(file_storage).convert("RGB")
    img_resized = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return arr, img


def predict_with_classical(features):
    results = {}
    for task, classes in TASK_CLASSES.items():
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
    if "deep" not in MODELS:
        return None
    model = MODELS["deep"]
    X = np.expand_dims(img_array, axis=0)
    try:
        preds = model.predict_proba(X)
        results = {}
        for task, classes in TASK_CLASSES.items():
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


def build_top_k(classical_results, deep_results, k=5):
    """Build top-k predictions list (from primary model, by confidence)."""
    source = deep_results if deep_results else classical_results
    if not source or "car_type" not in source:
        return []
    probs = source["car_type"].get("probabilities", {})
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])[:k]
    return [{"label": lbl, "confidence": float(conf)} for lbl, conf in sorted_probs]


def build_feedback(primary_results):
    """Build confidence-based feedback message and level."""
    if not primary_results or "car_type" not in primary_results:
        return {"level": "info", "message": "No prediction available", "color": "gray"}
    conf = primary_results["car_type"].get("confidence", 0.0)
    pred = primary_results["car_type"].get("prediction", "unknown")
    if conf >= 0.75:
        return {
            "level": "success",
            "message": f"High confidence: {pred} ({conf*100:.1f}%).",
            "color": "green",
        }
    if conf >= 0.5:
        return {
            "level": "warning",
            "message": f"Moderate confidence: {pred} ({conf*100:.1f}%). Maybe double-check.",
            "color": "orange",
        }
    return {
        "level": "error",
        "message": f"Low confidence: {pred} ({conf*100:.1f}%). Image is probably off-distribution.",
        "color": "red",
    }


def validate_upload(file):
    if not file or file.filename == "":
        return False, "No file selected", 400
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        return False, "Unsupported format. Use JPG, PNG, BMP, or WEBP.", 400
    try:
        img = Image.open(file)
        img.verify()
        file.stream.seek(0)
    except Exception:
        return False, "Invalid image file.", 400
    return True, None, 200


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "File too large. Max 16MB.",
        "code": 413,
    }), 413


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "success": False,
        "error": "Bad request - check the file you uploaded.",
        "code": 400,
    }), 400


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint.

    Returns JSON with:
    - prediction: primary predicted label (car_type)
    - confidence: confidence of primary prediction
    - top_k: top-k predictions with confidences
    - feedback: confidence-based feedback message
    - classical: per-task classical model results
    - deep: per-task deep model results (if available)
    - explanations: interpretable feature explanations
    - features: raw feature values
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400
    file = request.files["image"]
    ok, err_msg, status = validate_upload(file)
    if not ok:
        return jsonify({"success": False, "error": err_msg, "code": status}), status
    try:
        img_array, original_img = preprocess_image(file)
        features = extract_all_features(img_array)
        classical_results = predict_with_classical(features)
        deep_results = predict_with_deep(img_array)
        explanations = feature_importance_explanation(features)
        primary = deep_results if deep_results else classical_results
        primary_pred = primary.get("car_type", {}).get("prediction") if primary else None
        primary_conf = primary.get("car_type", {}).get("confidence", 0.0) if primary else 0.0
        top_k = build_top_k(classical_results, deep_results, k=5)
        feedback = build_feedback(primary)
        response = {
            "success": True,
            "prediction": primary_pred,
            "confidence": float(primary_conf),
            "top_k": top_k,
            "feedback": feedback,
            "classical": classical_results,
            "deep": deep_results,
            "explanations": explanations,
            "features": dict(zip(FEATURE_NAMES, features.tolist())),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "code": 500}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": len(MODELS),
        "loaded_models": list(MODELS.keys()),
    })


@app.route("/debug")
def debug():
    files = []
    if MODELS_DIR.exists():
        files = [{"name": f.name, "size": f.stat().st_size} for f in MODELS_DIR.iterdir() if f.is_file()]
    return jsonify({
        "models_dir": str(MODELS_DIR),
        "models_dir_exists": MODELS_DIR.exists(),
        "files": files,
        "loaded_models": list(MODELS.keys()),
    })


@app.route("/models")
def models_info():
    return jsonify({
        "loaded": list(MODELS.keys()),
        "tasks": {
            "car_type": CAR_TYPES,
            "door_count": DOOR_COUNTS,
            "seat_count": SEAT_COUNTS,
        },
    })


@app.route("/samples")
def samples():
    """List available sample images with metadata."""
    samples_dir = Path("static/samples")
    if not samples_dir.exists():
        return jsonify({"samples": []})
    files = sorted([f.name for f in samples_dir.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg")])
    out = []
    for fname in files:
        stem = Path(fname).stem.lower()
        if stem.startswith("sample_"):
            stem = stem[len("sample_"):]
        label = stem.replace("_", " ").title()
        out.append({
            "url": f"/static/samples/{fname}",
            "filename": fname,
            "label": label,
        })
    return jsonify({"samples": out})


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    load_models()
