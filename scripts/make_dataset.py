"""下载并准备 CompCars 数据集."""
import os
import json
from pathlib import Path
from scripts.data import RAW_DIR, METADATA_PATH, CAR_TYPES, DOOR_COUNTS, SEAT_COUNTS, SEED


def check_data_exists() -> bool:
    """检查数据是否已存在."""
    return (RAW_DIR / "part" / "attr.json").exists() and (RAW_DIR / "image").exists()


def print_instructions():
    """打印手动下载说明."""
    print("""
================================================================
  CompCars 数据集下载说明
================================================================

CompCars 数据集需要手动下载（非商业研究用途）。

1. 访问官网: http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
2. 下载以下文件:
   - web_data/image.zip (整车图片)
   - web_data/part.zip (属性数据)
3. 解压到以下目录结构:

   data/raw/compcars/
   ├── image/          <- 解压 image.zip
   │   ├── 1/          <- 车型ID文件夹
   │   │   ├── 001.jpg
   │   │   └── ...
   │   └── ...
   └── part/           <- 解压 part.zip
       ├── attr.json
       └── train_test_split

4. 运行验证:
   python -m scripts.data

环境变量:
   MAX_SAMPLES=5000  # 限制样本数（用于快速测试）

================================================================
""")


def create_metadata():
    """创建数据集元数据."""
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
    print(f"元数据已保存: {METADATA_PATH}")
    print(f"总图片数: {len(df):,}")
    for task, counts in metadata["class_counts"].items():
        print(f"\n{task} 分布:")
        for k, v in counts.items():
            print(f"  {k}: {v:,}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if check_data_exists():
        print("数据已存在，生成元数据...")
        create_metadata()
    else:
        print_instructions()


if __name__ == "__main__":
    main()
