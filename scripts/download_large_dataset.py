"""Download 10000+ real car photos via Bing Image Search (icrawler).

5 classes x 2000 images each. Saved to data/real_cars_large/<class>/.
"""
import os
import time
from pathlib import Path
from icrawler.builtin import BingImageCrawler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "real_cars_large"

CLASSES = {
    "sedan": ["sedan car exterior side view", "sedan car photo", "toyota camry sedan", "honda accord sedan", "bmw 3 series sedan"],
    "suv": ["suv car exterior side view", "suv car photo", "toyota rav4 suv", "honda cr-v suv", "tesla model y suv"],
    "mpv": ["mpv minivan car exterior", "minivan car photo side view", "toyota sienna minivan", "honda odyssey minivan", "kia carnival mpv"],
    "coupe": ["coupe car exterior side view", "coupe car photo", "ford mustang coupe", "audi a5 coupe", "bmw 4 series coupe"],
    "hatchback": ["hatchback car exterior side view", "hatchback car photo", "vw golf hatchback", "honda civic hatchback", "toyota corolla hatchback"],
}

PER_KEYWORD = 400
TOTAL_PER_CLASS = len(CLASSES["sedan"]) * PER_KEYWORD

def download_class(cls, keywords):
    cls_dir = OUT / cls
    cls_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(cls_dir.glob("*.jpg")))
    if existing >= 2000:
        print(f"  {cls}: already has {existing} images, skipping")
        return existing
    print(f"  {cls}: target {TOTAL_PER_CLASS} images, existing {existing}")
    for kw in keywords:
        print(f"    keyword: '{kw}' -> +{PER_KEYWORD}")
        crawler = BingImageCrawler(
            feeder_threads=2,
            parser_threads=2,
            downloader_threads=8,
            storage={"root_dir": str(cls_dir)},
            log_level=40,
        )
        try:
            crawler.crawl(
                keyword=kw,
                max_num=PER_KEYWORD,
                min_size=(200, 150),
                filters={"type": "photo"},
                file_idx_offset="auto",
            )
        except Exception as e:
            print(f"    error: {e}")
        time.sleep(1)
    final = len(list(cls_dir.glob("*.jpg")))
    print(f"  {cls}: {final} images downloaded")
    return final

def main():
    print("=" * 60)
    print("Downloading 10000+ real car photos")
    print("=" * 60)
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for cls, keywords in CLASSES.items():
        n = download_class(cls, keywords)
        total += n
    print(f"\nTotal: {total} images in {OUT}")
    for cls in CLASSES:
        n = len(list((OUT / cls).glob("*.jpg")))
        print(f"  {cls}: {n}")

if __name__ == "__main__":
    main()
