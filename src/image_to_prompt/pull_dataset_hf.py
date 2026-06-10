"""
VM.AI — Pull dataset from Hugging Face Hub.

Downloads the image dataset from HF and reconstructs
data/image_to_prompt/final/{train,val,test}/{label}/{filename}.jpg
plus the split CSVs.

Environment variables (from src/image_to_prompt/.env):
  HF_TOKEN           — Hugging Face API token (optional for public repos)
  HF_DATASET_REPO_ID — Dataset repository ID
"""

import csv
import shutil
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import os  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_DATASET_REPO_ID = os.environ["HF_DATASET_REPO_ID"]

FINAL = Path("data/image_to_prompt/final")


def main():
    print(f"Downloading dataset from {HF_DATASET_REPO_ID} ...")
    dataset = load_dataset(HF_DATASET_REPO_ID, token=HF_TOKEN)

    if FINAL.exists():
        print(f"Removing existing {FINAL} ...")
        shutil.rmtree(FINAL)

    for split_name in ["train", "val", "test"]:
        split = dataset[split_name]
        class_names = split.features["label"].names
        split_dir = FINAL / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, example in enumerate(split):
            label = class_names[example["label"]]
            img = example["image"]

            label_dir = split_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{label}_{i:05d}.jpg"
            save_path = label_dir / filename
            img.save(str(save_path), "JPEG")

            rows.append({"path": f"{split_name}/{label}/{filename}", "label": label})

        csv_path = FINAL / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(["path", "label"])
            for row in rows:
                writer.writerow([row["path"], row["label"]])

        print(f"  {split_name}: {len(split)} images -> {split_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
