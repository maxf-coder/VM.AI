"""
Model loader — pre-loads all AI models at backend startup
when LAZY_LOADING is disabled.
"""

import logging

from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from app.core.logging_config import setup_logging

logger = setup_logging()
_uvicorn_log = logging.getLogger("uvicorn")


def load_all_models():
    """Load all AI model services into memory eagerly."""
    _uvicorn_log.info("Pre-loading all AI models at startup...")

    from app.services.parser import Parser
    Parser.get_instance().load()
    _uvicorn_log.info("  Parser model loaded")

    from app.services.task_matcher import task_matcher
    _ = task_matcher.model
    _uvicorn_log.info("  Task matcher model loaded")

    from app.services.img_to_prompt import ImgToPrompt
    ImgToPrompt.get_instance().load()
    _uvicorn_log.info("  Image-to-prompt service loaded")

    from app.services.duration import duration_service
    duration_service.load()
    _uvicorn_log.info("  Duration predictor loaded")

    _uvicorn_log.info("All AI models loaded successfully")
