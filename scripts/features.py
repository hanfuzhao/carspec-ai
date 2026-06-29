"""Interpretable visual feature extraction: body proportions, contour curvature, color statistics, texture features.

These features are used for:
1. Input for Classical ML model (HOG+color+handcrafted features + Random Forest)
2. Auxiliary input for Deep model (concatenated after global features)
3. Provide interpretability (which visual features influenced predictions)
"""
import numpy as np
from PIL import Image


def extract_color_histogram(img, bins=8):
    """Extract HSV color histogram."""
    hsv = np.array(Image.fromarray((img * 255).astype(np.uint8)).convert("HSV"), dtype=np.float32)
    h_hist = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s_hist = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 256))[0]
    v_hist = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 256))[0]
    h_hist = h_hist / (h_hist.sum() + 1e-7)
    s_hist = s_hist / (s_hist.sum() + 1e-7)
    v_hist = v_hist / (v_hist.sum() + 1e-7)
    return np.concatenate([h_hist, s_hist, v_hist])


def extract_aspect_ratio(img):
    """Body width-to-height ratio (vehicles are usually wider than tall)."""
    h, w = img.shape[:2]
    return np.array([w / h], dtype=np.float32)


def extract_edge_density(img):
    """Edge density (Sobel operator)."""
    gray = np.mean(img, axis=2)
    gx = np.abs(np.gradient(gray, axis=1))
    gy = np.abs(np.gradient(gray, axis=0))
    edge = np.sqrt(gx ** 2 + gy ** 2)
    return np.array([edge.mean(), edge.std()], dtype=np.float32)


def extract_body_proportion(img):
    """Body proportion features: upper/lower brightness ratio, reflects body-to-window ratio."""
    h, w = img.shape[:2]
    upper = img[:h // 2].mean()
    lower = img[h // 2:].mean()
    return np.array([upper, lower, upper - lower], dtype=np.float32)


def extract_symmetry(img):
    """Left-right symmetry (vehicles are usually left-right symmetric)."""
    flipped = img[:, ::-1]
    diff = np.abs(img - flipped)
    return np.array([diff.mean()], dtype=np.float32)


def extract_hog_features(img, orientations=8, pixels_per_cell=(32, 32)):
    """Simplified HOG features."""
    try:
        from skimage.feature import hog
        gray = np.array(Image.fromarray((img * 255).astype(np.uint8)).convert("L"))
        features = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return np.array([features.mean(), features.std(), features.max()], dtype=np.float32)
    except ImportError:
        return np.zeros(3, dtype=np.float32)


def extract_texture_features(img):
    """Simple texture features: local binary pattern statistics."""
    gray = (np.mean(img, axis=2) * 255).astype(np.uint8)
    h, w = gray.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    center = gray[1:-1, 1:-1]
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
            lbp = (lbp << 1) | (shifted >= center)
    hist = np.histogram(lbp, bins=16, range=(0, 256))[0]
    hist = hist / (hist.sum() + 1e-7)
    return hist.astype(np.float32)


def extract_all_features(img):
    """Extract all interpretable features and concatenate into a vector."""
    return np.concatenate([
        extract_color_histogram(img, bins=8),
        extract_aspect_ratio(img),
        extract_edge_density(img),
        extract_body_proportion(img),
        extract_symmetry(img),
        extract_hog_features(img),
        extract_texture_features(img),
    ])


FEATURE_NAMES = [
    *[f"h_hist_{i}" for i in range(8)],
    *[f"s_hist_{i}" for i in range(8)],
    *[f"v_hist_{i}" for i in range(8)],
    "aspect_ratio",
    "edge_mean", "edge_std",
    "upper_brightness", "lower_brightness", "brightness_diff",
    "symmetry",
    "hog_mean", "hog_std", "hog_max",
    *[f"lbp_{i}" for i in range(16)],
]
FEATURE_DIM = len(FEATURE_NAMES)


def feature_importance_explanation(features, top_k=5):
    """Generate interpretable explanations based on feature values."""
    explanations = []
    h_hist = features[0:8]
    dominant_hue = int(np.argmax(h_hist))
    hue_names = ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Pink"]
    explanations.append(f"Dominant color: {hue_names[dominant_hue]} family (proportion {h_hist[dominant_hue]:.1%})")
    ar = features[24]
    if ar > 1.2:
        explanations.append(f"Aspect ratio {ar:.2f} -> leans toward SUV/MPV (wider)")
    else:
        explanations.append(f"Aspect ratio {ar:.2f} -> leans toward sedan/coupe (lower)")
    edge_mean = features[25]
    if edge_mean > 0.1:
        explanations.append(f"High edge density ({edge_mean:.3f}) -> complex body lines")
    else:
        explanations.append(f"Low edge density ({edge_mean:.3f}) -> simple body lines")
    sym = features[28]
    explanations.append(f"Left-right symmetry error {sym:.4f} -> {'symmetric (front view)' if sym < 0.05 else 'asymmetric (side view)'}")
    return explanations[:top_k]


if __name__ == "__main__":
    dummy = np.random.rand(224, 224, 3).astype(np.float32)
    feats = extract_all_features(dummy)
    print(f"Feature dimension: {len(feats)}")
    print(f"Number of feature names: {len(FEATURE_NAMES)}")
    print("\nInterpretable explanations:")
    for exp in feature_importance_explanation(feats):
        print(f"  - {exp}")
