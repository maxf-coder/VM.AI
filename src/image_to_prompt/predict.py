"""
VM.AI — Predict activity class from an image using trained EfficientNet-B4.

Usage:
    from predict import predict
    result = predict(image_pil)
    # => {"label": "cooking", "probability": 0.97, "results": {...}}
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent / "training"))
from train_classifier import build_model

_uvicorn_log = logging.getLogger("uvicorn")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_PATH = str(_PROJECT_ROOT / "models" / "efficientnet_b4_classifier" / "efficientnet_b4_classifier.pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASSES = [
    "basketball", "cleaning", "computer_work", "cooking", "cycling",
    "driving", "football", "gym", "office", "pet_care",
    "reading", "restaurant", "running", "shopping",
]

NUM_CLASSES = len(CLASSES)

val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_model_cache: Optional[dict] = None


def load_model(model_path: str = _DEFAULT_MODEL_PATH) -> None:
    """Eagerly load the model checkpoint into the global cache (used by model_loader)."""
    global _model_cache
    if _model_cache is not None and _model_cache["path"] == model_path:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model_cache = {
        "model": _load_model(model_path, device),
        "path": model_path,
    }


def _load_model(model_path: str, device: torch.device):
    """Load model checkpoint and return (model, device)."""
    _uvicorn_log.info("Loading image classifier model...")
    model = build_model(NUM_CLASSES).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    _uvicorn_log.info("Image classifier model loaded")
    return model


def predict(image: Image.Image, model_path: str = _DEFAULT_MODEL_PATH) -> dict:
    """
    Predict activity class from a PIL image.

    Args:
        image: PIL Image in RGB mode.
        model_path: Path to the .pth checkpoint.

    Returns:
        dict with:
            - label (str): predicted class name
            - probability (float): confidence of top prediction
            - results (dict): all class names mapped to their probabilities
    """
    global _model_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _model_cache is None or _model_cache["path"] != model_path:
        _model_cache = {
            "model": _load_model(model_path, device),
            "path": model_path,
        }

    model = _model_cache["model"]

    img = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    probs_np = probs.cpu().numpy()
    pred_idx = int(probs_np.argmax())

    results = {CLASSES[i]: round(float(probs_np[i]), 4) for i in range(NUM_CLASSES)}

    return {
        "label": CLASSES[pred_idx],
        "probability": round(float(probs_np[pred_idx]), 4),
        "results": results,
    }
