"""
VM.AI — Push the final/ dataset to Hugging Face Hub.

Walks final/{train,val,test}/{cat}/ directories, builds a DatasetDict,
and pushes to the configured HF dataset repo.

Environment variables (from src/image_to_prompt/.env):
  HF_TOKEN                — Hugging Face API token
  HF_DATASET_REPO_ID      — Dataset repository ID
  HF_DATASET_REPO_PRIVATE — Set to "false" for public repo (default: "true")
"""

import os
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, Image
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

HF_TOKEN = os.environ["HF_TOKEN"]
HF_DATASET_REPO_ID = os.environ["HF_DATASET_REPO_ID"]
HF_DATASET_REPO_PRIVATE = os.environ.get("HF_DATASET_REPO_PRIVATE", "true").lower() == "true"

FINAL = Path("data/image_to_prompt/final")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_split(split_name: str) -> Dataset:
    split_dir = FINAL / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing {split_dir}")

    images = []
    labels = []
    for cat_dir in sorted(split_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        for f in sorted(cat_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            images.append(str(f))
            labels.append(cat)

    class_names = sorted(set(labels))
    label_to_id = {n: i for i, n in enumerate(class_names)}
    label_ids = [label_to_id[l] for l in labels]

    ds = Dataset.from_dict({"image": images, "label": label_ids})
    ds = ds.cast_column("image", Image())
    ds = ds.cast_column("label", ClassLabel(names=class_names))
    return ds


def main():
    print(f"Loading splits from {FINAL} ...")

    train_ds = _load_split("train")
    val_ds = _load_split("val")
    test_ds = _load_split("test")

    dataset = DatasetDict({
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
    })

    print(f"  Train: {len(train_ds)}")
    print(f"  Val:   {len(val_ds)}")
    print(f"  Test:  {len(test_ds)}")

    print(f"\nPushing to {HF_DATASET_REPO_ID} (private={HF_DATASET_REPO_PRIVATE}) ...")
    dataset.push_to_hub(
        HF_DATASET_REPO_ID,
        token=HF_TOKEN,
        private=HF_DATASET_REPO_PRIVATE,
    )
    print("Done.")


if __name__ == "__main__":
    main()
