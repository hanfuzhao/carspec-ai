"""Synthetic data generator: generate simulated vehicle images for pipeline validation.

When the CompCars dataset is not downloaded, use this module to generate synthetic data.
Synthetic data simulates different vehicle attributes through different shapes, colors, and sizes.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

RAW_DIR = Path("data/raw/compcars")
IMG_DIR = RAW_DIR / "image"
PART_DIR = RAW_DIR / "part"
SEED = 42

CAR_TEMPLATES = {
    "sedan": {"body_h": 80, "body_w": 200, "roof_h": 50, "roof_w": 100, "roof_offset": 30},
    "suv": {"body_h": 110, "body_w": 200, "roof_h": 80, "roof_w": 130, "roof_offset": 20},
    "mpv": {"body_h": 120, "body_w": 210, "roof_h": 90, "roof_w": 160, "roof_offset": 15},
    "coupe": {"body_h": 70, "body_w": 190, "roof_h": 40, "roof_w": 80, "roof_offset": 40},
    "hatchback": {"body_h": 90, "body_w": 180, "roof_h": 60, "roof_w": 110, "roof_offset": 25},
}

ATTR_PROBS = {
    "sedan": {"door": {"2": 0.2, "4": 0.7, "5": 0.1}, "seat": {"2": 0.1, "5": 0.8, "7": 0.1}},
    "suv": {"door": {"2": 0.1, "4": 0.6, "5": 0.3}, "seat": {"2": 0.05, "5": 0.5, "7": 0.45}},
    "mpv": {"door": {"4": 0.3, "5": 0.7}, "seat": {"5": 0.3, "7": 0.7}},
    "coupe": {"door": {"2": 0.9, "4": 0.1}, "seat": {"2": 0.85, "5": 0.15}},
    "hatchback": {"door": {"2": 0.3, "4": 0.5, "5": 0.2}, "seat": {"2": 0.1, "5": 0.85, "7": 0.05}},
}

CAR_COLORS = [
    (220, 220, 220), (50, 50, 50), (180, 30, 30), (30, 80, 180),
    (30, 130, 50), (200, 200, 50), (150, 150, 150), (80, 80, 80),
]


def draw_car(img_size, car_type, color, door_count, seat_count):
    img = Image.new("RGB", (img_size, img_size), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    tpl = CAR_TEMPLATES[car_type]
    cx, cy = img_size // 2, img_size // 2
    bx1 = cx - tpl["body_w"] // 2
    by1 = cy - tpl["body_h"] // 2 + 20
    bx2 = cx + tpl["body_w"] // 2
    by2 = cy + tpl["body_h"] // 2 + 20
    draw.rounded_rectangle([bx1, by1, bx2, by2], radius=15, fill=color)
    rx1 = cx - tpl["roof_w"] // 2 + tpl["roof_offset"]
    ry1 = by1 - tpl["roof_h"]
    rx2 = rx1 + tpl["roof_w"]
    ry2 = by1 + 5
    draw.rounded_rectangle([rx1, ry1, rx2, ry2], radius=10, fill=tuple(min(c + 30, 255) for c in color))
    wx1 = rx1 + 10
    wy1 = ry1 + 10
    wx2 = rx2 - 10
    wy2 = ry2 - 5
    draw.rounded_rectangle([wx1, wy1, wx2, wy2], radius=5, fill=(100, 150, 200))
    wheel_r = 18
    wheel_y = by2 - 5
    for wx in [bx1 + 40, bx2 - 40]:
        draw.ellipse([wx - wheel_r, wheel_y - wheel_r, wx + wheel_r, wheel_y + wheel_r], fill=(30, 30, 30))
        draw.ellipse([wx - 8, wheel_y - 8, wx + 8, wheel_y + 8], fill=(150, 150, 150))
    n_doors = int(door_count)
    door_w = tpl["body_w"] / (n_doors + 1)
    for i in range(1, n_doors + 1):
        dx = bx1 + int(door_w * i)
        draw.line([dx, by1 + 10, dx, by2 - 10], fill=(80, 80, 80), width=2)
    noise = np.random.randint(-10, 10, (img_size, img_size, 3), dtype=np.int16)
    arr = np.array(img, dtype=np.int16) + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_synthetic_dataset(n_samples=6000, img_size=224):
    np.random.seed(SEED)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    car_types = list(CAR_TEMPLATES.keys())
    for i in range(n_samples):
        car_type = np.random.choice(car_types)
        color = CAR_COLORS[np.random.randint(len(CAR_COLORS))]
        door_probs = ATTR_PROBS[car_type]["door"]
        door_count = np.random.choice(list(door_probs.keys()), p=list(door_probs.values()))
        seat_probs = ATTR_PROBS[car_type]["seat"]
        seat_count = np.random.choice(list(seat_probs.keys()), p=list(seat_probs.values()))
        img = draw_car(img_size, car_type, color, door_count, seat_count)
        model_id = i % 100 + 1
        img_subdir = str(model_id)
        img_subdir_path = IMG_DIR / img_subdir
        img_subdir_path.mkdir(parents=True, exist_ok=True)
        img_name = f"{i:06d}.jpg"
        img_path = img_subdir_path / img_name
        img.save(img_path, quality=90)
        max_speed = np.random.randint(120, 300)
        displacement = np.random.choice([1.5, 2.0, 2.5, 3.0, 4.0])
        rows.append({
            "img_path": str(img_path),
            "model_id": model_id,
            "car_type": car_type,
            "door_count": door_count,
            "seat_count": seat_count,
            "max_speed": max_speed,
            "displacement": displacement,
        })
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{n_samples}")
    df = pd.DataFrame(rows)
    attr_data = []
    for _, row in df.iterrows():
        attr_data.append({
            "model_id": int(row["model_id"]),
            "max_speed": int(row["max_speed"]),
            "displacement": float(row["displacement"]),
            "num_doors": row["door_count"],
            "num_seats": row["seat_count"],
            "type": row["car_type"],
            "img_path": row["img_path"],
        })
    with open(PART_DIR / "attr.json", "w") as f:
        json.dump(attr_data, f)
    with open(PART_DIR / "train_test_split", "w") as f:
        for _, row in df.iterrows():
            rel_path = f"{row['model_id']}/{Path(row['img_path']).name}"
            is_test = 1 if np.random.random() < 0.2 else 0
            f.write(f"{rel_path} {row['model_id']} {is_test}\n")
    print(f"Synthetic dataset generation complete: {n_samples} images")
    print(f"Car type distribution:\n{df['car_type'].value_counts()}")
    return df


if __name__ == "__main__":
    n = int(os.environ.get("N_SAMPLES", "6000"))
    generate_synthetic_dataset(n_samples=n)
