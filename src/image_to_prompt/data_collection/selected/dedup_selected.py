"""
VM.AI — Remove within-category perceptual duplicates from selected/.

Uses same phash logic as analyze_dataset.py.
Keeps the first image per hash group, deletes the rest.
"""

from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image

SELECTED = Path("data/image_to_prompt/selected")
OUTLIERS_DIR = "outliers"  # directory name to exclude
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _is_image(f: Path) -> bool:
    return f.suffix.lower() in IMAGE_EXTENSIONS


def collect_images() -> dict[str, list[Path]]:
    images_by_cat: dict[str, list[Path]] = defaultdict(list)
    for cat_dir in sorted(SELECTED.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == OUTLIERS_DIR:
            continue
        cat = cat_dir.name
        for f in sorted(cat_dir.rglob("*")):
            if f.is_file() and _is_image(f):
                images_by_cat[cat].append(f)
    return dict(images_by_cat)


def main():
    if not SELECTED.is_dir():
        print("selected/ not found")
        return

    images_by_cat = collect_images()
    total_images = sum(len(v) for v in images_by_cat.values())
    print(f"Collected {total_images} images across {len(images_by_cat)} categories\n")

    hash_map: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    total = sum(len(v) for v in images_by_cat.values())
    processed = 0

    for cat, paths in sorted(images_by_cat.items()):
        for p in paths:
            try:
                h = str(imagehash.phash(Image.open(p)))
                hash_map[h].append((cat, p))
            except Exception:
                pass
            processed += 1
            if processed % 2000 == 0:
                print(f"    Hashed {processed}/{total}")

    total_kept = 0
    total_deleted = 0
    per_cat_kept: dict[str, int] = defaultdict(int)
    per_cat_deleted: dict[str, int] = defaultdict(int)

    for h, group in hash_map.items():
        if len(group) <= 1:
            cat = group[0][0]
            per_cat_kept[cat] += 1
            total_kept += 1
            continue

        cats_in_group = {c for c, _ in group}
        if len(cats_in_group) > 1:
            for cat, p in group:
                per_cat_kept[cat] += 1
                total_kept += 1
            continue

        cat = next(iter(cats_in_group))
        group.sort(key=lambda x: x[1])
        kept, *to_delete = group

        per_cat_kept[cat] += 1
        total_kept += 1

        for _, p in to_delete:
            p.unlink()
            per_cat_deleted[cat] += 1
            total_deleted += 1

    print(f"\nResults:")
    print(f"  Kept:   {total_kept}")
    print(f"  Deleted: {total_deleted}")
    print()
    for cat in sorted(per_cat_kept):
        kept = per_cat_kept[cat]
        deleted = per_cat_deleted.get(cat, 0)
        if deleted > 0:
            print(f"  {cat}: {kept} kept, {deleted} deleted")
        else:
            print(f"  {cat}: {kept} kept, 0 deleted")
    print("\nDone.")


if __name__ == "__main__":
    main()
