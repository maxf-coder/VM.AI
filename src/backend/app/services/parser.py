import logging
import os
import sys
from typing import Optional, Dict, Any

from app.schemas.nlp import NlpAddPayload, NlpPayloadField
from app.schemas.task import TaskPayload

logger = logging.getLogger(__name__)

# Allowed fields for modify output
ALLOWED_MODIFY_FIELDS = {
    "name",
    "start",
    "deadline",
    "difficulty",
    "duration",
    "category",
    "location",
    "importance",
    "fixed_time",
    "fixed_start",
}

# Fields to remove from add output (not in NlpAddPayload schema)
REMOVE_FIELDS = {"recurrent", "recurrence_days"}


def _setup_parser_path():
    """Add parser directory to path."""
    from pathlib import Path
    from app.core.logging_config import setup_logging
    
    parser_dir = Path(__file__).resolve().parent.parent.parent.parent / "parser"
    
    if not parser_dir.exists():
        logger = setup_logging()
        logger.warning(f"Parser directory not found: {parser_dir}")
        return
    
    if str(parser_dir) not in sys.path:
        sys.path.insert(0, str(parser_dir))


def _convert_value_to_type(field: str, value: Any) -> Any:
    """Convert string value to proper type based on field."""
    if value is None:
        return None

    # difficulty: string 0.0-1.0 -> float
    if field == "difficulty" and isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.5

    # importance: string 0.0-1.0 -> float
    if field == "importance" and isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.5

    # duration: string minutes -> int
    if field == "duration" and isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 30

    # category: string -> list of strings
    if field == "category":
        if isinstance(value, list):
            return value if value else []
        if isinstance(value, str):
            return [value] if value else []
        return []

    # location: keep as string (or None)
    if field == "location":
        return value if value else None

    # fixed_time: already bool
    if field == "fixed_time":
        return bool(value) if value is not None else False

    return value if value is not None else None


class Parser:
    """
    NLP Parser service using TaskPlannerPredictor.

    Provides two public methods:
    - parse_add(): Parse new task from prompt
    - parse_modify(): Apply changes to existing task
    """

    _instance: Optional["Parser"] = None
    _predictor = None

    def __init__(self):
        """Initialize the parser. Model loads lazily on first use."""
        pass

    def _load_model(self):
        """Load the TaskPlannerPredictor model."""
        if Parser._predictor is None:
            _setup_parser_path()
            from chat import TaskPlannerPredictor

            logger.info("Loading TaskPlannerPredictor...")
            Parser._predictor = TaskPlannerPredictor()
            logger.info("Parser model loaded successfully")

    def _ensure_loaded(self):
        """Ensure model is loaded before use."""
        if Parser._predictor is None:
            self._load_model()

    def load(self):
        """Eagerly load the T5 model into memory (used by model_loader)."""
        self._ensure_loaded()

    @classmethod
    def get_instance(cls) -> "Parser":
        """Get singleton instance of Parser."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def parse_add(self, prompt: str) -> Optional[NlpAddPayload]:
        """
        Parse a new task from a natural language prompt.

        Args:
            prompt: Natural language task description (e.g., "gym session")

        Returns:
            NlpAddPayload: Parsed task with {value, predicted} fields
            None on error
        """
        self._ensure_loaded()
        try:
            raw_output = self._predictor.predict_add(prompt)
            return self._convert_to_nlp_add_payload(raw_output)
        except Exception as e:
            logger.error(f"parse_add error: {e}")
            return None

    def parse_modify(
        self,
        existing_task: TaskPayload,
        change_prompt: str
    ) -> Optional[Dict[str, Any]]:
        """
        Apply changes to an existing task based on a change prompt.

        Args:
            existing_task: TaskPayload schema (current task)
            change_prompt: Natural language change instruction

        Returns:
            Dict with values only (e.g., {"importance": 0.95})
            None on error or invalid output
        """
        self._ensure_loaded()
        try:
            task_dict = self._task_payload_to_dict(existing_task)
            raw_output = self._predictor.predict_modify(task_dict, change_prompt)

            if "error" in raw_output:
                logger.warning(f"Parse modify error: {raw_output.get('error')}")
                return None

            return self._convert_modify_output(raw_output)
        except Exception as e:
            logger.error(f"parse_modify error: {e}")
            return None

    def _task_payload_to_dict(self, task: TaskPayload) -> Dict[str, Dict[str, Any]]:
        """Convert TaskPayload schema to dict with {value, predicted} format."""
        result = {}
        task_data = task.model_dump() if hasattr(task, "model_dump") else task.dict()

        for field, value in task_data.items():
            result[field] = {"value": value, "predicted": False}

        return result

    def _convert_to_nlp_add_payload(self, raw_output: dict) -> Optional[NlpAddPayload]:
        """Convert raw parser output to NlpAddPayload schema."""
        converted = {}

        for field, entry in raw_output.items():
            if field in REMOVE_FIELDS:
                continue

            if isinstance(entry, dict):
                value = entry.get("value")
                predicted = entry.get("predicted", True)
            else:
                value = entry
                predicted = True

            converted_value = _convert_value_to_type(field, value)

            converted[field] = NlpPayloadField(
                value=converted_value,
                predicted=predicted
            )

        try:
            return NlpAddPayload(**converted)
        except Exception as e:
            logger.error(f"NlpAddPayload validation error: {e}")
            return None

    def _convert_modify_output(
        self,
        raw_output: dict
    ) -> Optional[Dict[str, Any]]:
        """Convert raw modify output to simple key-value dict."""
        result = {}

        for field, entry in raw_output.items():
            if field not in ALLOWED_MODIFY_FIELDS:
                continue

            if isinstance(entry, dict):
                value = entry.get("value")
            else:
                value = entry

            converted_value = _convert_value_to_type(field, value)
            result[field] = converted_value

        return result if result else None


parser_service = Parser.get_instance()