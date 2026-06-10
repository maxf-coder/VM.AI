"""
VM.AI — Prepare & Validate Selected Images

Phase 0: Copy raw/ → selected/ (destroys & recreates selected/)
Phase 1: Flatten selected/<category>/<source>/... → selected/<category>/
Phase 2: Validate: remove corrupted, convert to JPG, remove too-small, deduplicate.
"""

import hashlib
import json
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

from PIL import Image

DATA_ROOT = Path("data/image_to_prompt")
RAW = DATA_ROOT / "raw"
SELECTED = DATA_ROOT / "selected"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
MIN_SIZE = 180

REPORT: dict = {
    "date": str(date.today()),
    "min_size": MIN_SIZE,
    "phase_0": {},
    "phase_1": {},
    "phase_2": {},
}


def _is_image(file: Path) -> bool:
    return file.suffix.lower() in IMAGE_EXTENSIONS


def _find_all_images(root: Path) -> list[Path]:
    return sorted([f for f in root.rglob("*") if f.is_file() and _is_image(f)])


def phase_0_prepare():
    print("=" * 60)
    print("Phase 0: Copy raw/ → selected/")
    print("=" * 60)

    if not RAW.is_dir():
        print("  raw/ directory not found — nothing to copy")
        print()
        return

    if SELECTED.exists():
        shutil.rmtree(SELECTED)
        print("  Deleted existing selected/")

    SELECTED.mkdir(parents=True, exist_ok=True)

    copied_categories = []
    for cat_dir in sorted(RAW.iterdir()):
        if not cat_dir.is_dir():
            continue
        dest = SELECTED / cat_dir.name
        shutil.copytree(cat_dir, dest)
        print(f"  {cat_dir.name}: copied")
        copied_categories.append(cat_dir.name)

    REPORT["phase_0"] = {
        "total_categories": len(copied_categories),
        "categories": copied_categories,
    }
    print(f"  Total categories copied: {len(copied_categories)}")
    print()


def phase_1_flatten():
    print("=" * 60)
    print("Phase 1: Flatten source subdirs")
    print("=" * 60)

    per_category = {}
    for cat_dir in sorted(SELECTED.iterdir()):
        if not cat_dir.is_dir():
            continue
        sources = sorted([d for d in cat_dir.iterdir() if d.is_dir()])
        if not sources:
            continue

        category = cat_dir.name
        moved = 0
        for src_dir in sources:
            source_name = src_dir.name
            images = _find_all_images(src_dir)
            for img in images:
                ext = img.suffix.lower()
                stem = img.stem
                new_name = f"{source_name}_{stem}{ext}"
                dest = cat_dir / new_name
                counter = 1
                while dest.exists():
                    dest = cat_dir / f"{source_name}_{stem}_{counter}{ext}"
                    counter += 1
                shutil.move(str(img), str(dest))
                moved += 1

            shutil.rmtree(src_dir, ignore_errors=True)

        per_category[category] = moved
        print(f"  {category}: {moved} images flattened")

    REPORT["phase_1"] = {"per_category_after_flatten": per_category}
    print()


def phase_2_validate():
    print("=" * 60)
    print("Phase 2: Validate images")
    print("=" * 60)

    cat_order = sorted(d.name for d in SELECTED.iterdir() if d.is_dir())

    stats = {}
    for cat in cat_order:
        cat_dir = SELECTED / cat
        count = len([f for f in cat_dir.iterdir() if f.is_file() and _is_image(f)])
        stats[cat] = {
            "initial": count,
            "corrupted": 0,
            "converted": 0,
            "too_small": 0,
            "dedup_conflicts": defaultdict(int),
            "dedup_total": 0,
        }

    all_hashes: dict[str, list[Path]] = {}

    for cat in cat_order:
        cat_dir = SELECTED / cat
        images = sorted([f for f in cat_dir.iterdir() if f.is_file() and _is_image(f)])

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception:
                img_path.unlink(missing_ok=True)
                stats[cat]["corrupted"] += 1
                continue

            if img_path.suffix.lower() not in (".jpg", ".jpeg"):
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        new_path = img_path.with_suffix(".jpg")
                        counter = 1
                        while new_path.exists():
                            new_path = img_path.with_suffix(f"_{counter}.jpg")
                            counter += 1
                        img.save(new_path, "JPEG", quality=95)
                    img_path.unlink(missing_ok=True)
                    img_path = new_path
                    stats[cat]["converted"] += 1
                except Exception:
                    img_path.unlink(missing_ok=True)
                    stats[cat]["corrupted"] += 1
                    continue

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                if w < MIN_SIZE or h < MIN_SIZE:
                    img_path.unlink(missing_ok=True)
                    stats[cat]["too_small"] += 1
                    continue
            except Exception:
                img_path.unlink(missing_ok=True)
                stats[cat]["corrupted"] += 1
                continue

            try:
                h = hashlib.sha256(img_path.read_bytes()).hexdigest()
                all_hashes.setdefault(h, []).append(img_path)
            except Exception:
                img_path.unlink(missing_ok=True)
                stats[cat]["corrupted"] += 1

    for h, paths in all_hashes.items():
        if len(paths) <= 1:
            continue

        keep_cat = paths[0].parent.name
        for dup in paths[1:]:
            dup_cat = dup.parent.name
            dup.unlink(missing_ok=True)
            stats[dup_cat]["dedup_total"] += 1
            if dup_cat == keep_cat:
                continue
            stats[dup_cat]["dedup_conflicts"][keep_cat] += 1

    grand_total_remaining = 0
    per_category_report = {}
    for cat in cat_order:
        cat_dir = SELECTED / cat
        s = stats[cat]
        remaining = len([f for f in cat_dir.iterdir() if f.is_file() and _is_image(f)])
        total_removed = s["corrupted"] + s["too_small"] + s["dedup_total"]

        per_category_report[cat] = {
            "initial": s["initial"],
            "corrupted": s["corrupted"],
            "converted": s["converted"],
            "too_small": s["too_small"],
            "dedup_total": s["dedup_total"],
            "dedup_conflicts": dict(s["dedup_conflicts"]),
            "remaining": remaining,
        }

        print(f"  {cat}: {s['initial']} → {remaining} (-{total_removed})")
        grand_total_remaining += remaining

    REPORT["phase_2"] = {
        "per_category": per_category_report,
        "grand_total_remaining": grand_total_remaining,
    }

    print(f"\n{'='*60}")
    print(f"Total remaining across all categories: {grand_total_remaining}")


def main():
    phase_0_prepare()
    phase_1_flatten()
    phase_2_validate()

    report_path = DATA_ROOT / "prepare_report.json"
    with open(report_path, "w") as f:
        json.dump(REPORT, f, indent=2)
    print(f"\nReport written to {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
