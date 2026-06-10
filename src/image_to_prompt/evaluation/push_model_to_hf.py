"""
VM.AI — Push trained model to Hugging Face Hub.

Uploads the .pth file, evaluation report, chart images,
and auto-generates a model card README.

Environment variables (from src/image_to_prompt/.env):
  HF_TOKEN               — Hugging Face API token
  HF_MODEL_REPO_ID       — Model repository ID
  HF_MODEL_REPO_PRIVATE  — Set to "false" for public repo (default: "true")
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(Path(__file__).parent.parent / ".env")

HF_TOKEN = os.environ["HF_TOKEN"]
HF_MODEL_REPO_ID = os.environ["HF_MODEL_REPO_ID"]
HF_MODEL_REPO_PRIVATE = os.environ.get("HF_MODEL_REPO_PRIVATE", "true").lower() == "true"

MODEL_DIR = Path("models") / "efficientnet_b4_classifier"
MODEL_PATH = MODEL_DIR / "efficientnet_b4_classifier.pth"
REPORT_PATH = MODEL_DIR / "evaluation_report.json"
ASSETS_DIR = Path("assets/image_classifier")

ASSETS_TO_UPLOAD = [
    (ASSETS_DIR / "confusion_matrix.png", "confusion_matrix.png"),
    (ASSETS_DIR / "per_class_metrics.png", "per_class_metrics.png"),
    (ASSETS_DIR / "topk_accuracy.png", "topk_accuracy.png"),
]


def build_readme(report: dict) -> str:
    top1 = report.get("top1_accuracy", "?")
    top3 = report.get("topk_accuracy", {}).get("top3", "?")
    macro_f1 = report.get("macro_f1", "?")
    weighted_f1 = report.get("weighted_f1", "?")
    test_samples = report.get("test_samples", "?")
    per_class = report.get("per_class", {})

    class_rows = ""
    for cn in sorted(per_class.keys()):
        m = per_class[cn]
        class_rows += (
            f"| {cn} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |\n"
        )

    has_charts = any(p.exists() for p, _ in ASSETS_TO_UPLOAD)

    header = """---
language: en
license: mit
tags:
  - image-classification
  - efficientnet
  - vm-ai
  - activity-recognition
datasets:
  - maxf-coder/task_image_classifier
metrics:
  - accuracy
  - f1
---

# VM.AI — Image Classifier

EfficientNet-B4 trained on 14 activity categories for the image-to-prompt pipeline.

## Performance

| Metric | Value |
|--------|-------|
| Test samples | {test_samples} |
| Top-1 accuracy | {top1} |
| Top-3 accuracy | {top3} |
| Macro F1 | {macro_f1} |
| Weighted F1 | {weighted_f1} |

## Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
{class_rows}
## Usage

```python
import torch
import timm
from PIL import Image
from torchvision import transforms

model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=14)
model.load_state_dict(torch.load("efficientnet_b4_classifier.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = Image.open("photo.jpg").convert("RGB")
tensor = transform(img).unsqueeze(0)
with torch.no_grad():
    logits = model(tensor)
pred = logits.argmax(1).item()
```

## Training

Two-phase training: 5 frozen epochs (head only) + 20 unfrozen epochs (last 2 blocks).
Optimizer: AdamW with cosine annealing. Mixed precision (AMP).
See [train_classifier.py](https://github.com/Infiteri/VM.AI) for details.
"""

    if has_charts:
        header += """
## Charts

![Confusion matrix](confusion_matrix.png)
![Per-class metrics](per_class_metrics.png)
![Top-K accuracy](topk_accuracy.png)
"""

    return header


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    report = {}
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            report = json.load(f)
        print(f"Loaded evaluation report ({len(report)} keys)")

    api = HfApi()
    api.create_repo(
        HF_MODEL_REPO_ID,
        private=HF_MODEL_REPO_PRIVATE,
        repo_type="model",
        exist_ok=True,
    )
    print(f"Repository ready: {HF_MODEL_REPO_ID}")

    api.upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo="efficientnet_b4_classifier.pth",
        repo_id=HF_MODEL_REPO_ID,
        token=HF_TOKEN,
    )
    print("  Uploaded efficientnet_b4_classifier.pth")

    if REPORT_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(REPORT_PATH),
            path_in_repo="evaluation_report.json",
            repo_id=HF_MODEL_REPO_ID,
            token=HF_TOKEN,
        )
        print("  Uploaded evaluation_report.json")

    for local_path, repo_path in ASSETS_TO_UPLOAD:
        if local_path.exists():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=HF_MODEL_REPO_ID,
                token=HF_TOKEN,
            )
            print(f"  Uploaded {repo_path}")

    readme = build_readme(report)
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO_ID,
        token=HF_TOKEN,
    )
    print("  Uploaded README.md (model card)")

    print(f"\nModel pushed to {HF_MODEL_REPO_ID}")
    print("Done.")


if __name__ == "__main__":
    main()
