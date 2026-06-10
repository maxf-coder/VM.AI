"""
VM.AI — Pull trained model from Hugging Face Hub.

Downloads the entire model repo to models/efficientnet_b4_classifier/.
Includes .pth, evaluation report, charts, and README.

Environment variables (from src/image_to_prompt/.env):
  HF_TOKEN           — Hugging Face API token (optional for public repos)
  HF_MODEL_REPO_ID   — Model repository ID
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv(Path(__file__).parent / ".env")

HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_MODEL_REPO_ID = os.environ["HF_MODEL_REPO_ID"]

MODEL_DIR = Path("models") / "efficientnet_b4_classifier"


def main():
    if MODEL_DIR.exists():
        print(f"Removing existing {MODEL_DIR} ...")
        shutil.rmtree(MODEL_DIR)

    print(f"Downloading model from {HF_MODEL_REPO_ID} ...")
    snapshot_download(
        repo_id=HF_MODEL_REPO_ID,
        local_dir=str(MODEL_DIR),
        token=HF_TOKEN,
        local_dir_use_symlinks=False,
    )
    print(f"Model pulled to {MODEL_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
