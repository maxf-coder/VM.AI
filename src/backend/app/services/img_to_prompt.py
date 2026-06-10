"""
Image-to-prompt service — extracts activity from an image and generates a task prompt.

Uses the trained EfficientNet-B4 model via predict.py.
Follows the same singleton + lazy-loading pattern as other services.

Usage:
    from app.services.img_to_prompt import ImgToPrompt
    result = ImgToPrompt.get_instance().classify(image_pil)
    # => {"prompt": "Running session on Monday at 10:00", "label": "running", "confidence": 0.97}
"""

import io
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from PIL import Image

from app.core.config import settings
from app.core.logging_config import setup_logging

logger = setup_logging()

# Add image_to_prompt to path for predict.py import
_image_to_prompt_dir = Path(__file__).resolve().parent.parent.parent.parent / "image_to_prompt"
if str(_image_to_prompt_dir) not in sys.path:
    sys.path.insert(0, str(_image_to_prompt_dir))

CONFIDENCE_THRESHOLD = 0.5

PROMPT_TEMPLATES = {
    "running": "Running session on {day} at {time}",
    "cycling": "Cycling session on {day} at {time}",
    "cooking": "Cooking some food on {day} at {time}",
    "restaurant": "Go to restaurant on {day} at {time}",
    "shopping": "Go shopping on {day} at {time}",
    "office": "Office work on {day} at {time}",
    "football": "Football training on {day} at {time}",
    "cleaning": "Cleaning on {day} at {time}",
    "driving": "Drive the car on {day} at {time}",
    "reading": "Read a book on {day} at {time}",
    "computer_work": "Work on the computer on {day} at {time}",
    "basketball": "Basketball training on {day} at {time}",
    "pet_care": "Take care of my pet on {day} at {time}",
    "gym": "Gym session on {day} at {time}",
}

DEFAULT_TEMPLATE = "Occupied time on {day} at {time}"


EXIF_DATETIME_TAGS = [36867, 36868, 306]  # DateTimeOriginal, DateTimeDigitized, DateTime


def _parse_exif_datetime(image: Image.Image) -> Optional[datetime]:
    """Try multiple EXIF datetime tags, return first found."""
    try:
        exif_data = image.getexif()
        if exif_data is None:
            logger.info("No EXIF data found in image")
            return None
        for tag in EXIF_DATETIME_TAGS:
            raw_str = exif_data.get(tag)
            if raw_str is not None:
                logger.info(f"EXIF tag {tag} found: {raw_str}")
                return datetime.strptime(raw_str, "%Y:%m:%d %H:%M:%S")
        logger.info(f"No EXIF datetime tag found (tried tags: {EXIF_DATETIME_TAGS})")
        return None
    except Exception as e:
        logger.warning(f"Failed to parse EXIF datetime: {e}")
        return None


def _round_to_5min(dt: datetime) -> str:
    """Round datetime to nearest 5-minute interval and return HH:MM string."""
    minutes = dt.minute
    rounded = round(minutes / 5) * 5
    if rounded == 60:
        dt = dt + timedelta(hours=1)
        dt = dt.replace(minute=0)
    else:
        dt = dt.replace(minute=rounded)
    result = dt.strftime("%H:%M")
    logger.info(f"Rounded time: {dt.strftime('%H:%M')} → {result}")
    return result


def _generate_prompt(label: str, day: str, time_str: str) -> str:
    """Generate a prompt string from label using the template map."""
    template = PROMPT_TEMPLATES.get(label, DEFAULT_TEMPLATE)
    return template.format(day=day, time=time_str)


class ImgToPrompt:
    """
    Image-to-prompt service.

    Extracts EXIF datetime, classifies the image, and generates a task prompt.
    Model loads lazily on first call (via predict.py).
    """

    _instance: Optional["ImgToPrompt"] = None

    def __init__(self):
        """Initialize the service."""
        pass

    @classmethod
    def get_instance(cls) -> "ImgToPrompt":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        """Eagerly load the image classifier model into memory (used by model_loader)."""
        from predict import load_model as _load_clf
        _load_clf(settings.CLASSIFIER_MODEL_PATH)

    def classify(self, image: Image.Image) -> dict:
        """
        Classify an image and generate a task prompt.

        Args:
            image: PIL Image (any format, will be converted to RGB).

        Returns:
            dict with:
                - prompt (str): natural language task description
                - label (str): predicted activity class (or "unknown")
                - confidence (float): prediction probability
        """
        logger.info("Starting image classification...")

        # Convert to RGB (handles RGBA, P, etc.)
        image = image.convert("RGB")
        logger.info(f"Image converted to RGB: {image.size[0]}x{image.size[1]}")

        # Extract EXIF datetime (tries multiple tags)
        dt = _parse_exif_datetime(image)
        if dt is not None:
            day = dt.strftime("%A")
            time_str = _round_to_5min(dt)
            logger.info(f"EXIF parsed: day={day}, time={time_str}")
        else:
            dt = datetime.now()
            logger.info(f"No EXIF datetime, using current server time: {dt}")
            day = dt.strftime("%A")
            time_str = _round_to_5min(dt)
            logger.info(f"Server time parsed: day={day}, time={time_str}")

        # Classify via predict.py
        logger.info("Running model prediction...")
        from predict import predict
        result = predict(image, settings.CLASSIFIER_MODEL_PATH)
        label = result["label"]
        confidence = result["probability"]
        logger.info(f"Model prediction: label='{label}', confidence={confidence:.4f}")

        # Apply threshold
        if confidence < CONFIDENCE_THRESHOLD:
            logger.info(f"Confidence {confidence:.3f} < threshold {CONFIDENCE_THRESHOLD}, setting label to 'unknown'")
            label = "unknown"

        # Generate prompt
        prompt = _generate_prompt(label, day, time_str)
        logger.info(f"Generated prompt: '{prompt}'")

        return {
            "prompt": prompt,
            "label": label,
            "confidence": confidence,
        }


img_to_prompt = ImgToPrompt.get_instance()