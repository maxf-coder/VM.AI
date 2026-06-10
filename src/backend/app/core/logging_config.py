import logging
import sys
from pathlib import Path

# 1. Correct the path: parent.parent.parent goes to 'backend' folder
# This resolves to: C:\VM.AI\src\backend\logs
LOG_DIR = Path(__file__).parent.parent.parent / "logs"

# 2. Create the folder if it doesn't exist (parents=True ensures the path is built)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "backend.log"

def setup_logging():
    """Configures the application-wide logger."""
    logger = logging.getLogger("vm_ai_backend")
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler: Saves EVERYTHING to the file
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console Handler: Shows only WARNING+ to keep terminal clean
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
