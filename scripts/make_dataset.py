"""Download and prepare the CompCars dataset."""
import os
import json
from pathlib import Path
from scripts.data import RAW_DIR, METADATA_PATH, CAR_TYPES, DOOR_COUNTS, SEAT_COUNTS, SEED


def check_data_exists() -> bool:
    """Check if data already exists."""
    return (RAW_DIR / "part" / "attr.json").exists() and (RAW_DIR / "image").exists()


def print_instructions():
    """Print manual download instructions."""
    print("""
================================================================
  CompCars Dataset Download Instructions
================================================================

The CompCars dataset needs to be downloaded manually (for non-commercial research use).

1. Visit the official website: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
2. Download the following files:
   - web_data/image.zip (vehicle images)
   - web_data/part.zip (attribute data)
3. Extract to the following directory structure:

   data/raw/compcars/
   ├── image/          <- Extract image.zip
   │   ├── 1/          <- Model ID folder
   │   │   ├── 001.jpg
   │   │   └── ...
   │   └── ...
   └── part/           <- Extract part.zip
       ├── attr.json
       └── train_test_split

4. Run validation:
   python -m scripts.data

Environment variables:
   MAX_SAMPLES=5000  # Limit the number of samples (for quick testing)

================================================================
""")


def create_metadata():
    """Create dataset metadata."""
    from scripts.data import load_image_labels
    df = load_image_labels()
    metadata = {
        "dataset": "CompCars (Comprehensive Cars)",
        "source": "http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/",
        "paper": "Yang et al., A Large-Scale Car Dataset for Fine-Grained Categorization and Verification, CVPR 2015",
        "tasks": {
            "car_type": CAR_TYPES,
            "door_count": DOOR_COUNTS,
            "seat_count": SEAT_COUNTS,
        },
        "n_total": int(len(df)),
        "class_counts": {
            "car_type": df["car_type"].value_counts().to_dict(),
            "door_count": df["door_count"].value_counts().to_dict(),
            "seat_count": df["seat_count"].value_counts().to_dict(),
        },
        "seed": SEED,
    }
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"Metadata saved: {METADATA_PATH}")
    print(f"Total images: {len(df):,}")
    for task, counts in metadata["class_counts"].items():
        print(f"\n{task} distribution:")
        for k, v in counts.items():
            print(f"  {k}: {v:,}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if check_data_exists():
        print("Data already exists, generating metadata...")
        create_metadata()
    else:
        print_instructions()


if __name__ == "__main__":
    main()
