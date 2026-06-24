"""CompCars dataset loading and preprocessing."""
import os
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

RAW_DIR = Path("data/raw/compcars")
PART_DIR = RAW_DIR / "part"
IMG_DIR = RAW_DIR / "image"
SPLIT_DIR = Path("data/processed")
METADATA_PATH = RAW_DIR / "metadata.json"
SEED = 42

CAR_TYPES = ["sedan", "suv", "mpv", "coupe", "hatchback"]
TYPE2ID = {name: i for i, name in enumerate(CAR_TYPES)}
ID2TYPE = {i: name for i, name in enumerate(CAR_TYPES)}

DOOR_COUNTS = ["2", "4", "5"]
DOOR2ID = {name: i for i, name in enumerate(DOOR_COUNTS)}
ID2DOOR = {i: name for i, name in enumerate(DOOR_COUNTS)}

SEAT_COUNTS = ["2", "5", "7"]
SEAT2ID = {name: i for i, name in enumerate(SEAT_COUNTS)}
ID2SEAT = {i: name for i, name in enumerate(SEAT_COUNTS)}

TASKS = {
    "car_type": {"labels": CAR_TYPES, "id2label": ID2TYPE, "label2id": TYPE2ID},
    "door_count": {"labels": DOOR_COUNTS, "id2label": ID2DOOR, "label2id": DOOR2ID},
    "seat_count": {"labels": SEAT_COUNTS, "id2label": ID2SEAT, "label2id": SEAT2ID},
}

IMG_SIZE = 224
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0"))


def _map_car_type(raw_type: str) -> str:
    """Map CompCars raw car type to 5 classes."""
    t = str(raw_type).lower().strip()
    if any(k in t for k in ["sedan", "saloon"]):
        return "sedan"
    if any(k in t for k in ["suv", "jeep", "cross"]):
        return "suv"
    if any(k in t for k in ["mpv", "minivan", "van"]):
        return "mpv"
    if any(k in t for k in ["coupe", "convertible", "roadster"]):
        return "coupe"
    if any(k in t for k in ["hatchback", "hatch"]):
        return "hatchback"
    return "sedan"


def _map_door_count(num_doors) -> str:
    """Map door count to 3 classes."""
    try:
        n = int(float(num_doors))
    except (ValueError, TypeError):
        return "4"
    if n <= 3:
        return "2"
    if n == 5:
        return "5"
    return "4"


def _map_seat_count(num_seats) -> str:
    """Map seat count to 3 classes."""
    try:
        n = int(float(num_seats))
    except (ValueError, TypeError):
        return "5"
    if n <= 3:
        return "2"
    if n >= 6:
        return "7"
    return "5"


def load_attributes() -> pd.DataFrame:
    """Load CompCars attribute data."""
    attr_path = PART_DIR / "attr.json"
    if not attr_path.exists():
        raise FileNotFoundError(
            f"Attribute file does not exist: {attr_path}. Please run make_dataset.py to download or manually place the data first."
        )
    with open(attr_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for item in data:
        model_id = item.get("model_id")
        max_speed = item.get("max_speed", 0)
        displacement = item.get("displacement", 0)
        num_doors = item.get("num_doors", 4)
        num_seats = item.get("num_seats", 5)
        car_type = item.get("type", "sedan")
        rows.append({
            "model_id": model_id,
            "max_speed": max_speed,
            "displacement": displacement,
            "num_doors": _map_door_count(num_doors),
            "num_seats": _map_seat_count(num_seats),
            "car_type": _map_car_type(car_type),
        })
    return pd.DataFrame(rows)


def load_image_labels() -> pd.DataFrame:
    """Load image label data and associate attributes."""
    label_path = PART_DIR / "train_test_split"
    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file does not exist: {label_path}. Please run make_dataset.py to download or manually place the data first."
        )
    attr_df = load_attributes()
    attr_by_id = attr_df.set_index("model_id").to_dict("index")
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rel_path = parts[0]
            model_id = int(parts[1])
            is_test = int(parts[2])
            if model_id not in attr_by_id:
                continue
            attr = attr_by_id[model_id]
            img_path = IMG_DIR / rel_path
            if not img_path.exists():
                continue
            rows.append({
                "img_path": str(img_path),
                "model_id": model_id,
                "is_test": is_test,
                "car_type": attr["car_type"],
                "door_count": attr["num_doors"],
                "seat_count": attr["num_seats"],
                "max_speed": attr["max_speed"],
                "displacement": attr["displacement"],
            })
    df = pd.DataFrame(rows)
    if MAX_SAMPLES > 0 and len(df) > MAX_SAMPLES:
        df = df.groupby("car_type", group_keys=False).apply(
            lambda g: g.sample(min(MAX_SAMPLES // len(CAR_TYPES), len(g)), random_state=SEED)
        ).reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """Stratified split of train/validation/test sets by car type."""
    rng = np.random.RandomState(seed)
    train, val, test = ([], [], [])
    for _, grp in df.groupby("car_type"):
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * 0.1))
        n_val = int(round(n * 0.1))
        test.append(grp.loc[idx[:n_test]])
        val.append(grp.loc[idx[n_test:n_test + n_val]])
        train.append(grp.loc[idx[n_test + n_val:]])
    out = [
        pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for parts in (train, val, test)
    ]
    return (out[0], out[1], out[2])


def get_splits(force: bool = False):
    """Get data splits with cache."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {s: SPLIT_DIR / f"{s}.csv" for s in ("train", "val", "test")}
    if all(p.exists() for p in paths.values()) and not force:
        return tuple(pd.read_csv(paths[s]) for s in ("train", "val", "test"))
    df = load_image_labels()
    train, val, test = make_splits(df)
    for s, d in zip(("train", "val", "test"), (train, val, test)):
        d.to_csv(paths[s], index=False)
    return (train, val, test)


def load_image(path: str, size: int = IMG_SIZE):
    """Load and preprocess a single image."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def image_generator(df: pd.DataFrame, batch_size=32, size=IMG_SIZE, shuffle=True, seed=SEED):
    """Image batch generator for deep learning training."""
    rng = np.random.RandomState(seed)
    n = len(df)
    indices = np.arange(n)
    while True:
        if shuffle:
            rng.shuffle(indices)
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_imgs = []
            batch_car_type = []
            batch_door = []
            batch_seat = []
            for idx in batch_idx:
                row = df.iloc[idx]
                img = load_image(row["img_path"], size)
                batch_imgs.append(img)
                batch_car_type.append(TYPE2ID[str(row["car_type"])])
                batch_door.append(DOOR2ID[str(row["door_count"])])
                batch_seat.append(SEAT2ID[str(row["seat_count"])])
            yield (
                np.array(batch_imgs),
                {
                    "car_type": np.array(batch_car_type),
                    "door_count": np.array(batch_door),
                    "seat_count": np.array(batch_seat),
                },
            )


if __name__ == "__main__":
    df = load_image_labels()
    print(f"Loaded {len(df):,} images")
    print("\nCar type distribution:")
    print(df["car_type"].value_counts())
    print("\nDoor count distribution:")
    print(df["door_count"].value_counts())
    print("\nSeat count distribution:")
    print(df["seat_count"].value_counts())
