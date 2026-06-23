"""可解释视觉特征提取：车身比例、轮廓曲率、颜色统计、纹理特征.

这些特征用于：
1. Classical ML 模型的输入（HOG+颜色+手工特征+SVM）
2. Deep 模型的辅助输入（拼接在全局特征后）
3. 提供可解释性（哪些视觉特征影响了预测）
"""
import numpy as np
from PIL import Image


def extract_color_histogram(img, bins=8):
    """提取HSV颜色直方图."""
    hsv = np.array(Image.fromarray((img * 255).astype(np.uint8)).convert("HSV"), dtype=np.float32)
    h_hist = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s_hist = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 256))[0]
    v_hist = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 256))[0]
    h_hist = h_hist / (h_hist.sum() + 1e-7)
    s_hist = s_hist / (s_hist.sum() + 1e-7)
    v_hist = v_hist / (v_hist.sum() + 1e-7)
    return np.concatenate([h_hist, s_hist, v_hist])


def extract_aspect_ratio(img):
    """车身宽高比（车辆通常宽>高）."""
    h, w = img.shape[:2]
    return np.array([w / h], dtype=np.float32)


def extract_edge_density(img):
    """边缘密度（Sobel算子）."""
    gray = np.mean(img, axis=2)
    gx = np.abs(np.gradient(gray, axis=1))
    gy = np.abs(np.gradient(gray, axis=0))
    edge = np.sqrt(gx ** 2 + gy ** 2)
    return np.array([edge.mean(), edge.std()], dtype=np.float32)


def extract_body_proportion(img):
    """车身比例特征：上半/下半亮度比，反映车身与车窗比例."""
    h, w = img.shape[:2]
    upper = img[:h // 2].mean()
    lower = img[h // 2:].mean()
    return np.array([upper, lower, upper - lower], dtype=np.float32)


def extract_symmetry(img):
    """左右对称性（车辆通常左右对称）."""
    flipped = img[:, ::-1]
    diff = np.abs(img - flipped)
    return np.array([diff.mean()], dtype=np.float32)


def extract_hog_features(img, orientations=8, pixels_per_cell=(32, 32)):
    """简化版HOG特征."""
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
        # 降维：取统计量
        return np.array([features.mean(), features.std(), features.max()], dtype=np.float32)
    except ImportError:
        return np.zeros(3, dtype=np.float32)


def extract_texture_features(img):
    """简单纹理特征：局部二值模式统计."""
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
    """提取全部可解释特征，拼接为向量."""
    return np.concatenate([
        extract_color_histogram(img, bins=8),       # 24维
        extract_aspect_ratio(img),                   # 1维
        extract_edge_density(img),                   # 2维
        extract_body_proportion(img),                # 3维
        extract_symmetry(img),                       # 1维
        extract_hog_features(img),                   # 3维
        extract_texture_features(img),               # 16维
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
    """根据特征值生成可解释说明."""
    explanations = []
    # 颜色特征
    h_hist = features[0:8]
    dominant_hue = int(np.argmax(h_hist))
    hue_names = ["红", "橙", "黄", "绿", "青", "蓝", "紫", "粉"]
    explanations.append(f"主色调: {hue_names[dominant_hue]}色系 (占比{h_hist[dominant_hue]:.1%})")
    # 宽高比
    ar = features[24]
    if ar > 1.2:
        explanations.append(f"宽高比 {ar:.2f} → 偏向SUV/MPV（较宽）")
    else:
        explanations.append(f"宽高比 {ar:.2f} → 偏向轿车/跑车（较矮）")
    # 边缘密度
    edge_mean = features[25]
    if edge_mean > 0.1:
        explanations.append(f"边缘密度高 ({edge_mean:.3f}) → 车身线条复杂")
    else:
        explanations.append(f"边缘密度低 ({edge_mean:.3f}) → 车身线条简洁")
    # 对称性
    sym = features[28]
    explanations.append(f"左右对称性误差 {sym:.4f} → {'对称（正面视角）' if sym < 0.05 else '不对称（侧面视角）'}")
    return explanations[:top_k]


if __name__ == "__main__":
    # 测试
    dummy = np.random.rand(224, 224, 3).astype(np.float32)
    feats = extract_all_features(dummy)
    print(f"特征维度: {len(feats)}")
    print(f"特征名称数: {len(FEATURE_NAMES)}")
    print("\n可解释说明:")
    for exp in feature_importance_explanation(feats):
        print(f"  - {exp}")
