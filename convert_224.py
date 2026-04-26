"""
Pre-resize VeRi images and save to VeRi224.

  image_train  → 252×252  (= 224 × 1.125, exact upscale target of Random2DTranslation)
                           Random2DTranslation then just does RandomCrop(224) with zero resize cost.
  image_query  → 224×224
  image_test   → 224×224  (T.Resize in test pipeline becomes a no-op)

Usage:
    python convert_224.py                                      # local defaults
    python convert_224.py --src /content/VeRi --dst /content/VeRi224  # Colab

After running, veri.py dataset_dir is already set to "VeRi224".
"""

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm

QUALITY = 95
WORKERS = 8

SPLIT_SIZES = {
    "image_train": (252, 252),  # 224 * 1.125 — lets RandomCrop replace Random2DTranslation
    "image_query": (224, 224),
    "image_test":  (224, 224),
}


def convert(src_path: Path, dst_path: Path, size: tuple) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    img.save(dst_path, "JPEG", quality=QUALITY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path(r"C:\Users\soura\Code\2026\reid\data\VeRi"))
    parser.add_argument("--dst", type=Path, default=Path(r"C:\Users\soura\Code\2026\reid\data\VeRi224"))
    args = parser.parse_args()

    tasks = []
    for split, size in SPLIT_SIZES.items():
        for src_path in (args.src / split).glob("*.jpg"):
            tasks.append((src_path, args.dst / split / src_path.name, size))

    print(f"Converting {len(tasks)} images  {args.src} → {args.dst} ...")
    print(f"  image_train → 252×252,  image_query/test → 224×224")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(convert, s, d, sz): s for s, d, sz in tasks}
        with tqdm(total=len(tasks), unit="img") as bar:
            for f in as_completed(futures):
                f.result()
                bar.update(1)

    print(f"\nDone. Output: {args.dst}")
    for split in SPLIT_SIZES:
        n = len(list((args.dst / split).glob("*.jpg")))
        print(f"  {split}: {n} images")


if __name__ == "__main__":
    main()
