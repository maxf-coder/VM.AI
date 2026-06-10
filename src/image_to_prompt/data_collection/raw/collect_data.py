"""
VM.AI — Image Data Collection Script

Downloads images for the image-to-prompt classifier from multiple sources.
Sources: Open Images V7 (via fiftyone), Kaggle (via kagglehub), Pixabay (via API)

Usage:
    uv run python src/image_to_prompt/data_collection/collect_data.py
"""

import hashlib
import json
import os
import shutil
import random
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
import fiftyone as fo
import fiftyone.zoo as foz
import pandas as pd
import requests

load_dotenv(Path(__file__).parent / ".env")

DATA_ROOT = Path("data/image_to_prompt/raw")

PIXABAY_API_KEY = os.getenv("PIXABAY_API_KEY", "YOUR_KEY")
PIXABAY_BASE_URL = "https://pixabay.com/api/"

CATEGORIES = {
    "running": {
        "sources": [
            {
                "type": "kaggle_csv",
                "dataset": "meetnagadia/human-action-recognition-har-dataset",
                "csv_path": "Human Action Recognition/Training_set.csv",
                "filename_col": "filename",
                "label_col": "label",
                "filter_value": "running",
                "image_root": "Human Action Recognition/train",
                "target": 800,
            },
            {
                "type": "pixabay",
                "keywords": ["person running"],
                "target": 400,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "lumierebatalong/human-activity-recognition-train-and-test-folders",
                "subfolder": "HAR/train/running",
                "target": 840,
            },
        ],
    },
    "cycling": {
        "sources": [
            {
                "type": "kaggle_csv",
                "dataset": "meetnagadia/human-action-recognition-har-dataset",
                "csv_path": "Human Action Recognition/Training_set.csv",
                "filename_col": "filename",
                "label_col": "label",
                "filter_value": "cycling",
                "image_root": "Human Action Recognition/train",
                "target": 800,
            },
            {
                "type": "pixabay",
                "keywords": ["cycling bicycle"],
                "target": 400,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "lumierebatalong/human-activity-recognition-train-and-test-folders",
                "subfolder": "HAR/train/cycling",
                "target": 840,
            },
        ],
    },
    "cooking": {
        "sources": [
            {
                "type": "openimages",
                "labels": [
                    "Gas stove", "Frying pan", "Cutting board", "Wok",
                    "Cooking spray", "Kitchen utensil", "Kitchenware",
                    "Slow cooker", "Pressure cooker", "Mixing bowl",
                ],
                "target": 700,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "dataclusterlabs/kitchen-full-image-dataset",
                "subfolder": "",
                "target": 400,
            },
            {
                "type": "pixabay",
                "keywords": ["kitchen", "cooking", "cookware", "kitchenware", "chef stove"],
                "target": 500,
            },
        ],
    },
    "restaurant": {
        "sources": [
            {
                "type": "openimages",
                "labels": [
                    "Fast food", "Kitchen & dining room table",
                    "Tableware", "Coffee", "Wine",
                ],
                "target": 400,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "kmader/food41",
                "subfolder": "images",
                "balanced": True,
                "target": 900,
            },
            {
                "type": "pixabay",
                "keywords": ["restaurant", "cafe", "restaurant inside"],
                "target": 300,
            },
        ],
    },
    "shopping": {
        "sources": [
            {
                "type": "openimages",
                "labels": ["Convenience store", "Cart", "Plastic bag", "Handbag"],
                "target": 300,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "humansintheloop/supermarket-shelves-dataset",
                "subfolder": "Supermarket shelves/Supermarket shelves/images",
                "target": 45,
            },
            {
                "type": "pixabay",
                "keywords": ["grocery store", "mall", "clothes store"],
                "target": 1100,
            },
        ],
    },
    "office": {
        "sources": [
            {
                "type": "openimages",
                "labels": [
                    "Office building", "Office supplies", "Computer monitor",
                    "Whiteboard", "Filing cabinet", "Printer",
                ],
                "target": 500,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "sordi-ai/office-dataset",
                "subfolder": "images",
                "target": 500,
            },
            {
                "type": "pixabay",
                "keywords": ["office", "office room", "office desk"],
                "target": 500,
            },
        ],
    },
    "football": {
        "sources": [
            {
                "type": "openimages",
                "labels": ["Football"],
                "target": 100,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "ligtfeather/football-vs-rugby-image-classification",
                "subfolder": "input/train/soccer",
                "target": 900,
            },
            {
                "type": "pixabay",
                "keywords": ["football"],
                "target": 300,
            },
        ],
    },
    "cleaning": {
        "sources": [
            {
                "type": "openimages",
                "labels": ["Washing machine", "Sink", "Soap dispenser"],
                "target": 300,
            },
            {
                "type": "pixabay",
                "keywords": ["cleaning", "person cleaning house", "mopping floor", "washing dishes"],
                "target": 1000,
            },
        ],
    },
    "driving": {
        "sources": [
            {
                "type": "openimages",
                "labels": ["Car", "Seat belt", "Land vehicle", "Taxi"],
                "target": 400,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "rightway11/state-farm-distracted-driver-detection",
                "subfolder": "imgs/train/c0",
                "target": 600,
            },
            {
                "type": "pixabay",
                "keywords": ["person driving car", "car"],
                "target": 200,
            },
        ],
    },
    "reading": {
        "sources": [
            {
                "type": "openimages",
                "labels": ["Book", "Bookcase"],
                "target": 150,
            },
            {
                "type": "pixabay",
                "keywords": ["person reading book", "reading", "reading on the sofa", "reading library", "book", "library"],
                "target": 2000,
            },
        ],
    },
    "computer_work": {
        "sources": [
            {
                "type": "openimages",
                "labels": ["Computer monitor", "Computer keyboard", "Laptop", "Computer mouse"],
                "target": 400,
            },
            {
                "type": "pixabay",
                "keywords": ["person and laptop", "developer coding", "work in laptop"],
                "target": 1200,
            },
        ],
    },
    "basketball": {
        "sources": [
            {
                "type": "kaggle_subfolder",
                "dataset": "rishikeshkonapure/sports-image-dataset",
                "subfolder": "data/basketball",
                "target": 486,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "gpiosenka/sports-classification",
                "subfolder": "train/basketball",
                "target": 169,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "ponrajsubramaniian/sportclassificationdataset",
                "subfolder": "Sports-Type-Classifier-master/data/basketball",
                "target": 495,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "mmoreaux/caltech256",
                "subfolder": "256_ObjectCategories/256_ObjectCategories/006/basketball-hoop",
                "target": 90,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "sheikhzaib/sports-image-image-classification",
                "subfolder": "sports/basketball",
                "target": 486,
            },
            {
                "type": "pixabay",
                "keywords": ["basketball", "basketball field", "basketball player"],
                "target": 1000,
            },
        ],
    },
    "pet_care": {
        "sources": [
            {
                "type": "openimages",
                "labels": ["Dog", "Cat", "Dog bed", "Cat furniture"],
                "target": 400,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "tongpython/cat-and-dog",
                "subfolder": "training_set/Training_set/cats",
                "target": 350,
            },
            {
                "type": "kaggle_subfolder",
                "dataset": "tongpython/cat-and-dog",
                "subfolder": "training_set/training_set/dogs",
                "target": 350,
            },
            {
                "type": "pixabay",
                "keywords": ["pet", "person walking dog"],
                "target": 200,
            },
        ],
    },
    "gym": {
        "sources": [
            {
                "type": "openimages",
                "labels": [
                    "Dumbbell", "Treadmill", "Indoor rower",
                    "Stationary bicycle", "Training bench",
                    "Punching bag", "Horizontal bar",
                ],
                "target": 400,
            },
            {
                "type": "kaggle",
                "dataset": "hasyimabdillah/workoutexercises-images",
                "samples_per_folder": 32,
                "target": 700,
            },
            {
                "type": "pixabay",
                "keywords": ["gym workout"],
                "target": 200,
            },
        ],
    },
}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def handle_openimages(category: str, source: dict) -> dict:
    labels = source["labels"]
    target = source["target"]
    per_label = max(1, target // len(labels))
    out_dir = DATA_ROOT / category / "openimages"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [openimages] Loading up to ~{per_label} per label (from train+val): {labels}")

    results = {}
    for label in labels:
        label_slug = label.replace(" ", "_").lower()
        combined_dir = out_dir / label_slug
        combined_dir.mkdir(parents=True, exist_ok=True)

        total = 0
        for split in ("train", "validation"):
            ds_name = f"oi_{split}_{label_slug}"
            try:
                dataset = foz.load_zoo_dataset(
                    "open-images-v7",
                    split=split,
                    label_types="detections",
                    classes=[label],
                    max_samples=max(1, per_label // 2) if split == "validation" else per_label,
                    dataset_name=ds_name,
                )
            except Exception as e:
                print(f"    {label} ({split}): SKIP - {e}")
                continue

            if len(dataset) == 0:
                continue

            split_dir = combined_dir / split
            dataset.export(
                export_dir=str(split_dir),
                dataset_type=fo.types.ImageDirectory,
            )
            total += len(dataset)
            print(f"    {label} ({split}): {len(dataset)} images")

        results[label] = total
        print(f"    {label} total: {total} images")

    grand_total = sum(results.values())
    print(f"  [openimages] Total: {grand_total} images")
    return {"type": "openimages", "labels": labels, "per_label_results": results, "downloaded": grand_total}


def handle_kaggle(category: str, source: dict) -> dict:
    import kagglehub

    dataset_slug = source["dataset"]
    samples_per = source["samples_per_folder"]
    out_dir = DATA_ROOT / category / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [kaggle] Downloading dataset '{dataset_slug}'...")
    download_path = Path(kagglehub.dataset_download(dataset_slug))

    subfolders = sorted([
        d for d in download_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    print(f"  [kaggle] Found {len(subfolders)} subfolders")

    total = 0
    folder_results = {}
    for folder in subfolders:
        images = sorted([
            f for f in folder.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])
        if not images:
            continue
        sampled = random.sample(images, min(samples_per, len(images)))
        folder_out = out_dir / folder.name.replace(" ", "_").lower()
        folder_out.mkdir(parents=True, exist_ok=True)
        for img in sampled:
            shutil.copy2(img, folder_out / img.name)
        count = len(sampled)
        folder_results[folder.name] = count
        total += count
        print(f"    {folder.name}: {count}/{len(images)} images sampled")

    print(f"  [kaggle] Total: {total} images")
    return {"type": "kaggle", "dataset": dataset_slug, "per_folder": folder_results, "downloaded": total}


def handle_kaggle_csv(category: str, source: dict) -> dict:
    import kagglehub

    dataset = source["dataset"]
    csv_path = source["csv_path"]
    filename_col = source["filename_col"]
    label_col = source["label_col"]
    filter_value = source["filter_value"]
    image_root = source["image_root"]
    target = source["target"]
    out_dir = DATA_ROOT / category / "kaggle_csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [kaggle_csv] Downloading dataset '{dataset}'...")
    download_path = Path(kagglehub.dataset_download(dataset))

    csv_full = download_path / csv_path
    print(f"  [kaggle_csv] Reading {csv_full}...")
    df = pd.read_csv(csv_full)

    matched = df[df[label_col] == filter_value].head(target)
    print(f"  [kaggle_csv] Found {len(matched)} images for '{filter_value}'")

    total = 0
    for _, row in matched.iterrows():
        src = download_path / image_root / str(row[filename_col])
        if src.exists():
            shutil.copy2(src, out_dir / src.name)
            total += 1
        else:
            print(f"    MISSING: {src}")

    print(f"  [kaggle_csv] Copied {total} images")
    return {
        "type": "kaggle_csv",
        "dataset": dataset,
        "filter_value": filter_value,
        "matched": len(matched),
        "copied": total,
        "downloaded": total,
    }


def handle_kaggle_subfolder(category: str, source: dict) -> dict:
    import kagglehub

    dataset = source["dataset"]
    subfolder = source.get("subfolder", "")
    target = source["target"]
    balanced = source.get("balanced", False)
    out_dir = DATA_ROOT / category / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [kaggle_subfolder] Downloading dataset '{dataset}'...")
    download_path = Path(kagglehub.dataset_download(dataset))
    source_dir = (download_path / subfolder) if subfolder else download_path

    if not source_dir.exists():
        print(f"  [kaggle_subfolder] ERROR: subfolder '{subfolder}' not found")
        return {"type": "kaggle_subfolder", "dataset": dataset, "error": "subfolder not found", "downloaded": 0}

    total = 0
    results = {}

    if balanced:
        subdirs = sorted([d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
        per_folder = max(1, target // len(subdirs))
        print(f"  [kaggle_subfolder] Balanced mode: {len(subdirs)} subdirs, {per_folder} each")
        for sd in subdirs:
            images = sorted([f for f in sd.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
            if not images:
                continue
            sampled = random.sample(images, min(per_folder, len(images)))
            sub_out = out_dir / f"{dataset.replace('/', '_')}_{sd.name.replace(' ', '_').lower()}"
            sub_out.mkdir(parents=True, exist_ok=True)
            for img in sampled:
                shutil.copy2(img, sub_out / img.name)
            results[sd.name] = len(sampled)
            total += len(sampled)
            print(f"    {sd.name}: {len(sampled)}/{len(images)} images")
    else:
        images = sorted([f for f in source_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])
        if not images:
            print(f"  [kaggle_subfolder] No images found in '{subfolder}'")
            return {"type": "kaggle_subfolder", "dataset": dataset, "downloaded": 0}

        sampled = random.sample(images, min(target, len(images)))
        slug = subfolder.replace(" ", "_").replace("/", "_").lower() if subfolder else dataset.replace("/", "_")
        sub_out = out_dir / slug
        sub_out.mkdir(parents=True, exist_ok=True)
        for img in sampled:
            shutil.copy2(img, sub_out / img.name)
        results[subfolder] = len(sampled)
        total = len(sampled)
        print(f"    Sampled {len(sampled)}/{len(images)} images from '{subfolder or '(root)'}'")

    print(f"  [kaggle_subfolder] Total: {total} images")
    return {"type": "kaggle_subfolder", "dataset": dataset, "subfolder": subfolder, "per_folder": results, "downloaded": total}


def handle_pixabay(category: str, source: dict) -> dict:
    keywords = source["keywords"]
    target = source["target"]
    out_dir = DATA_ROOT / category / "pixabay"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_keyword = max(1, target // len(keywords))
    results = {}
    total = 0

    for kw in keywords:
        page = 1
        kw_total = 0
        while kw_total < per_keyword:
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
                print("    Rate limit low — waiting 60s...")
                time.sleep(60)

            data = resp.json()
            hits = data.get("hits", [])
            if not hits:
                break

            for photo in hits:
                img_url = photo.get("largeImageURL") or photo["webformatURL"].replace("_640", "_960")
                try:
                    img_data = requests.get(img_url, timeout=10).content
                    fname = f"{total:04d}.jpg"
                    (out_dir / fname).write_bytes(img_data)
                    total += 1
                    kw_total += 1
                except Exception:
                    continue
                if kw_total >= per_keyword:
                    break

            page += 1
            time.sleep(0.5)

        results[kw] = kw_total
        print(f"  [pixabay] '{kw}': {kw_total} images")
        if total >= target:
            break

    print(f"  [pixabay] Total: {total} images")
    return {"type": "pixabay", "keywords": list(results.keys()), "per_keyword": results, "downloaded": total}


SOURCE_HANDLERS = {
    "openimages": handle_openimages,
    "kaggle": handle_kaggle,
    "kaggle_csv": handle_kaggle_csv,
    "kaggle_subfolder": handle_kaggle_subfolder,
    "pixabay": handle_pixabay,
}


def write_metadata(category: str, source_results: list[dict]):
    meta = {
        "category": category,
        "date": str(date.today()),
        "sources": source_results,
        "total_raw": sum(s.get("downloaded", 0) for s in source_results),
    }
    meta_path = DATA_ROOT / category / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [metadata] Written to {meta_path}")


def process_category(category: str, config: dict):
    print(f"\n{'='*60}")
    print(f"  Category: {category}")
    print(f"{'='*60}")

    source_results = []
    for source in config["sources"]:
        stype = source["type"]
        handler = SOURCE_HANDLERS.get(stype)
        if handler is None:
            print(f"  [skip] Unknown source type: {stype}")
            continue
        try:
            result = handler(category, source)
            source_results.append(result)
        except Exception as e:
            print(f"  [error] {stype}: {e}")
            source_results.append({"type": stype, "error": str(e), "downloaded": 0})

    write_metadata(category, source_results)
    total = sum(s.get("downloaded", 0) for s in source_results)
    print(f"  Total raw for '{category}': {total}")


def main():
    import sys

    random.seed(42)

    categories_to_run = sys.argv[1:] if len(sys.argv) > 1 else list(CATEGORIES.keys())

    print("VM.AI — Image Data Collection")
    print(f"Data root: {DATA_ROOT.resolve()}")
    print(f"Categories ({len(categories_to_run)}): {categories_to_run}")

    for category in categories_to_run:
        if category not in CATEGORIES:
            print(f"Unknown category: {category}, skipping")
            continue
        process_category(category, CATEGORIES[category])

    print(f"\n{'='*60}")
    print("  Done! All categories processed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
