"""Pre-download HuggingFace models so they are cached locally."""
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
]


def download_models():
    for model_id in MODELS:
        for attempt in range(3):
            try:
                logger.info("Downloading %s (attempt %d/3)...", model_id, attempt + 1)
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(model_id)
                logger.info("  %s ready", model_id)
                break
            except Exception as e:
                logger.warning("  Attempt %d failed: %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    logger.error("  Failed to download %s after 3 attempts", model_id)
                    return False
    return True


if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
