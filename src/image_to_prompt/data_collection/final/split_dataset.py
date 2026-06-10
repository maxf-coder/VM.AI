"""
VM.AI — Split selected/ into train/val/test and write CSVs.

1. Collect all images from selected/<category>/**/*
2. Shuffle, cap at 1100 per category
3. Split 70/15/15 into final/train, final/val, final/test
4. Write train.csv, val.csv, test.csv
"""

import random
import shutil
from pathlib import Path

SELECTED = Path("data/image_to_prompt/selected")
FINAL = Path("data/image_to_prompt/final")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_PER_CATEGORY = 700
SEED = 42


def _is_image(file: Path) -> bool:
    return file.suffix.lower() in IMAGE_EXTENSIONS


def main():
    random.seed(SEED)

    if not SELECTED.is_dir():
        print("selected/ not found — nothing to split")
        return

    if FINAL.exists():
        shutil.rmtree(FINAL)
        print("Deleted existing final/")

    categories = sorted(d.name for d in SELECTED.iterdir() if d.is_dir())

    csv_rows = {split: [] for split in RATIOS}

    for cat in categories:
        cat_dir = SELECTED / cat
        images = sorted(
            [f for f in cat_dir.rglob("*") if f.is_file() and _is_image(f)]
        )
        if not images:
            print(f"  {cat}: no images found")
            continue

        random.shuffle(images)

        if len(images) > MAX_PER_CATEGORY:
            before = len(images)
            images = images[:MAX_PER_CATEGORY]
            print(f"  {cat}: {before} -> capped to {MAX_PER_CATEGORY}")

        total = len(images)
        train_end = int(total * RATIOS["train"])
        val_end = train_end + int(total * RATIOS["val"])

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, split_images in splits.items():
            split_dir = FINAL / split_name / cat
            split_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                dest = split_dir / img_path.name
                counter = 1
                while dest.exists():
                    stem = img_path.stem
                    dest = split_dir / f"{stem}_{counter}{img_path.suffix}"
                    counter += 1
                shutil.copy2(img_path, dest)
                csv_rows[split_name].append(f'"{split_name}/{cat}/{dest.name}","{cat}"')

        print(f"  {cat}: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test")

    for split_name, rows in csv_rows.items():
        csv_path = FINAL / f"{split_name}.csv"
        with open(csv_path, "w") as f:
            f.write('"path","label"\n')
            for row in rows:
                f.write(row + "\n")
        print(f"\n  {split_name}.csv: {len(rows)} rows")

    total_all = sum(len(r) for r in csv_rows.values())
    print(f"\nTotal images in final/: {total_all}")
    print("Done.")


if __name__ == "__main__":
    main()
