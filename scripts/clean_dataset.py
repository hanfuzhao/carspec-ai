"""Clean downloaded dataset: remove duplicates, corrupted images, non-car images.

Steps:
1. Validate each image can be opened
2. Remove exact duplicates by MD5 hash
3. Remove images that are too small or wrong aspect ratio
4. Report final count per class
"""
import os
import hashlib
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "real_cars_large"
MIN_SIZE = (200, 150)
MIN_ASPECT = 0.5
MAX_ASPECT = 3.0


def md5_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def clean_class(cls_dir):
    removed = {"corrupted": 0, "duplicate": 0, "too_small": 0, "bad_aspect": 0}
    valid = []
    seen_hashes = set()

    for img_path in sorted(cls_dir.glob("*.jpg")):
        try:
            img = Image.open(img_path)
            img.verify()
            img = Image.open(img_path)
            w, h = img.size
            if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
                removed["too_small"] += 1
                img_path.unlink()
                continue
            aspect = w / h
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                removed["bad_aspect"] += 1
                img_path.unlink()
                continue
            h_md5 = md5_file(img_path)
            if h_md5 in seen_hashes:
                removed["duplicate"] += 1
                img_path.unlink()
                continue
            seen_hashes.add(h_md5)
            valid.append(img_path)
        except Exception:
            removed["corrupted"] += 1
            try:
                img_path.unlink()
            except Exception:
                pass

    return valid, removed


def main():
    print("=" * 60)
    print("Cleaning dataset")
    print("=" * 60)
    total_valid = 0
    for cls_dir in sorted(DATA_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        before = len(list(cls_dir.glob("*.jpg")))
        valid, removed = clean_class(cls_dir)
        after = len(valid)
        total_valid += after
        print(f"  {cls_dir.name}: {before} -> {after} (removed: {removed})")

    print(f"\nTotal valid images: {total_valid}")


if __name__ == "__main__":
    main()
