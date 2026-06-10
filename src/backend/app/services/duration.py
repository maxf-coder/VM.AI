import sys
from pathlib import Path
from typing import Optional

from app.core.logging_config import setup_logging

logger = setup_logging()

_parser_dir = Path(__file__).resolve().parent.parent.parent.parent / "parser"
if str(_parser_dir) not in sys.path:
    sys.path.insert(0, str(_parser_dir))


class DurationService:
    _instance: Optional["DurationService"] = None
    _predictor = None

    def __init__(self):
        pass

    @classmethod
    def get_instance(cls) -> "DurationService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        if DurationService._predictor is None:
            from duration_predictor import DurationPredictor
            logger.info("Loading DurationPredictor...")
            DurationService._predictor = DurationPredictor()
            logger.info("DurationPredictor loaded")

    @property
    def predictor(self):
        if DurationService._predictor is None:
            self.load()
        return DurationService._predictor

    def predict(
        self,
        difficulty: float,
        importance: float,
        scheduled_duration: int,
        category: str,
        location: str,
        fixed_time: str = "",
        time_difference: float = -1,
    ) -> int:
        return self.predictor.predict(
            difficulty=difficulty,
            importance=importance,
            scheduled_duration=scheduled_duration,
            category=category,
            location=location,
            fixed_time=fixed_time,
            time_difference=time_difference,
        )


duration_service = DurationService.get_instance()
