"""
VM.AI — Download additional images to selected/{category}/new/

Downloads from Pixabay (and optionally Kaggle) into selected/{category}/new/
with per-image validation matching prepare_data.py requirements.
Any image that is corrupted, too small, or a duplicate is discarded
and the next image is fetched instead.
"""

import hashlib
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv(Path(__file__).parent / ".env")

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "YOUR_KEY")
PIXABAY_BASE_URL = "https://pixabay.com/api/"

DATA_ROOT = Path("data/image_to_prompt")
SELECTED = DATA_ROOT / "selected"
MIN_SIZE = 180
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}

CONFIG = {
    "cleaning": {
        "sources": [
            {
                "type": "pixabay",
                "keywords": [
                    "laundry detergent",
                    "dirty clothes",
                    "cleaning gloves",
                ],
                "target": 200,
            },
        ],
    },
    "shopping": {
        "sources": [
            {
                "type": "pixabay",
                "keywords": [
                    "supermarket",
                    "wallet",
                ],
                "target": 200,
            },
        ],
    },
}


def _is_image(file: Path) -> bool:
    return file.suffix.lower() in IMAGE_EXTENSIONS


def load_all_hashes() -> set[str]:
    all_h = set()
    for f in SELECTED.rglob("*"):
        if f.is_file() and _is_image(f):
            try:
                all_h.add(hashlib.sha256(f.read_bytes()).hexdigest())
            except Exception:
                continue
    return all_h


def validate_and_hash(data: bytes, existing_hashes: set[str], tmp_dir: Path, tag: str) -> tuple[bool, str, Path | None]:
    tmp_path = tmp_dir / f".tmp_{tag}"
    try:
        tmp_path.write_bytes(data)

        with Image.open(tmp_path) as img:
            img.verify()

        with Image.open(tmp_path) as img:
            img = img.convert("RGB")

        w, h = img.size
        if w < MIN_SIZE or h < MIN_SIZE:
            tmp_path.unlink(missing_ok=True)
            return False, "", None

        img.save(tmp_path, "JPEG", quality=95)

        h = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
        if h in existing_hashes:
            tmp_path.unlink(missing_ok=True)
            return False, "", None

        return True, h, tmp_path

    except Exception:
        tmp_path.unlink(missing_ok=True)
        return False, "", None


def download_pixabay_to_new(category: str, keywords: list[str], target: int, existing_hashes: set[str], new_dir: Path, start_count: int):
    per_keyword = max(1, target // len(keywords)) + 50
    downloaded = start_count
    skipped = 0
    tag_counter = 0

    for kw in keywords:
        page = 1
        kw_downloaded = 0
        target_reached = False

        while kw_downloaded < per_keyword and not target_reached:
            resp = requests.get(PIXABAY_BASE_URL, params={
                "key": PIXABAY_API_KEY,
                "q": kw,
                "image_type": "photo",
                "orientation": "horizontal",
                "safesearch": "true",
                "per_page": 200,
                "page": page,
                "min_width": 380,
                "min_height": 380,
            })
            remaining = int(resp.headers.get("X-RateLimit-Remaining", 100))
            if remaining < 5:
                print(f"    Rate limit low — waiting 60s...")
                time.sleep(60)

            data = resp.json()
            hits = data.get("hits", [])
            if not hits:
                break

            for photo in hits:
                if downloaded >= start_count + target:
                    target_reached = True
                    break
                if kw_downloaded >= per_keyword:
                    break

                img_url = photo.get("largeImageURL") or photo["webformatURL"].replace("_640", "_960")
                try:
                    img_data = requests.get(img_url, timeout=10).content
                except Exception:
                    skipped += 1
                    continue

                tag = f"{category}_{tag_counter}"
                tag_counter += 1
                valid, h, tmp_path = validate_and_hash(img_data, existing_hashes, new_dir, tag)

                if not valid:
                    skipped += 1
                    continue

                existing_hashes.add(h)
                fname = f"{downloaded:04d}.jpg"
                tmp_path.rename(new_dir / fname)
                downloaded += 1
                kw_downloaded += 1

            page += 1
            time.sleep(0.5)

        if target_reached:
            break

        print(f"    '{kw}': {kw_downloaded} new images")

    new_count = downloaded - start_count
    print(f"  [pixabay] Downloaded: {new_count}, Skipped: {skipped}")
    return new_count


def download_kaggle_to_new(category: str, dataset: str, subfolder: str, target: int, existing_hashes: set[str], new_dir: Path, start_count: int):
    import kagglehub

    print(f"    Downloading Kaggle dataset '{dataset}'...")
    path = kagglehub.dataset_download(dataset)
    source_dir = Path(path)
    if subfolder:
        source_dir = source_dir / subfolder

    images = sorted([f for f in source_dir.rglob("*") if f.is_file() and _is_image(f)])
    random.shuffle(images)

    downloaded = start_count
    skipped = 0
    tag_counter = 0

    for img_path in images:
        if downloaded >= start_count + target:
            break

        data = img_path.read_bytes()
        tag = f"{category}_kaggle_{tag_counter}"
        tag_counter += 1
        valid, h, tmp_path = validate_and_hash(data, existing_hashes, new_dir, tag)

        if not valid:
            skipped += 1
            continue

        existing_hashes.add(h)
        fname = f"{downloaded:04d}.jpg"
        tmp_path.rename(new_dir / fname)
        downloaded += 1

    new_count = downloaded - start_count
    print(f"  [kaggle] Copied: {new_count}, Skipped: {skipped}")
    return new_count


def process_category(category: str, config: dict):
    print(f"\n{'='*60}")
    print(f"  Category: {category}")
    print(f"{'='*60}")

    new_dir = SELECTED / category / "new"
    new_dir.mkdir(parents=True, exist_ok=True)

    existing_files = sorted([f for f in new_dir.iterdir() if f.is_file() and _is_image(f)])
    start_count = len(existing_files)

    print(f"  Loading hashes from selected/...")
    all_hashes = load_all_hashes()
    print(f"  Existing unique images across all categories: {len(all_hashes)}")
    print(f"  Already in new/: {start_count}")

    total_downloaded = 0
    for source in config["sources"]:
        stype = source["type"]
        target = source["target"]

        if stype == "pixabay":
            downloaded = download_pixabay_to_new(
                category, source["keywords"], target, all_hashes, new_dir, start_count + total_downloaded
            )
        elif stype == "kaggle_subfolder":
            downloaded = download_kaggle_to_new(
                category, source["dataset"], source["subfolder"], target, all_hashes, new_dir, start_count + total_downloaded
            )
        else:
            print(f"  [skip] Unknown source type: {stype}")
            continue

        total_downloaded += downloaded

    print(f"  Total new images for '{category}': {total_downloaded}")


def main():
    import sys

    random.seed(42)

    categories_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIG.keys())

    for cat in categories_to_run:
        if cat not in CONFIG:
            print(f"  [skip] Unknown category: {cat}")
            continue
        process_category(cat, CONFIG[cat])

    print("\nDone.")


if __name__ == "__main__":
    main()
