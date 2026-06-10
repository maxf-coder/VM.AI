"""
Task Matching Testing Script

Tests task matching for 20 diverse task names.
Logs input, output (including associated_task_name from DB).

Logs saved to C:\\VM.AI\\src\\backend\\logs\\task_matching_{YYYYMMDD}.log

Run from backend directory:
    cd src/backend
    python tests/test_task_matching.py
"""

import sys
import os
import json
from datetime import datetime
from uuid import UUID

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.services.task_matcher import TaskMatcher
from app.models.statistics import TaskStatistics


LOG_DIR = "logs"


class TestLogger:
    def __init__(self, test_type: str):
        self.test_type = test_type
        self.log_file = None
        self._open_log()

    def _open_log(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"task_matching_{date_str}.log"
        filepath = os.path.join(LOG_DIR, filename)
        self.log_file = open(filepath, "w", encoding="utf-8")

    def write(self, content: str):
        self.log_file.write(content)

    def write_json(self, title: str, data: dict):
        self.write(f"\n--- {title} ---\n")
        self.write(json.dumps(data, indent=2, default=str) + "\n")

    def close(self):
        if self.log_file:
            self.log_file.close()


def serialize_for_json(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    return obj


def run_task_matching_tests():
    print("Running task matching tests...")
    print(f"Logs will be saved to {LOG_DIR}/ directory\n")

    logger = TestLogger("task_matching")
    logger.write("=" * 70 + "\n")
    logger.write("TASK MATCHING TESTS (20 task names)\n")
    logger.write("=" * 70 + "\n\n")

    task_names = [
        "chemistry homework",
        "math assignment",
        "do chemistry",
        "math problems",
        "physics work",
        "bio report",
        "history paper",
        "morning jog",
        "buy groceries",
        "pay electricity bill",
        "dentist appointment",
        "finish project report",
        "call mom",
        "read book chapter",
        "clean the kitchen",
        "water the plants",
        "team meeting",
        "review pull request",
        "update resume",
        "yoga session",
        "send invoice",
        "send invoice to client",
        "pick up dry cleaning",
        "write blog post",
        "schedule oil change",
        "meditate for 10 minutes",
        "backup computer files",
        "register for online course",
    ]

    matcher = TaskMatcher()
    db = SessionLocal()

    logger.write(f"Total task names to test: {len(task_names)}\n")

    try:
        for i, task_name in enumerate(task_names, 1):
            logger.write(f"\n[Case {i}] Testing: '{task_name}'\n")

            logger.write_json("INPUT task_name", task_name)

            result = matcher.find_match(db, task_name)

            associated_task_name = None
            if result.get("associated_id"):
                stats = (
                    db.query(TaskStatistics)
                    .filter(TaskStatistics.id == result["associated_id"])
                    .first()
                )
                if stats:
                    associated_task_name = stats.task_name

            output = {
                "associated_id": result.get("associated_id"),
                "association_status": result.get("association_status"),
                "name_vector": result.get("name_vector"),
                "associated_task_name": associated_task_name,
            }
            output = serialize_for_json(output)

            logger.write_json("OUTPUT result", output)

            logger.write("\n    SUCCESS\n")

    except Exception as e:
        error_data = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        logger.write_json("ERROR", error_data)

    finally:
        db.close()

    logger.close()

    print("ALL TESTS COMPLETED")
    print(f"See logs in {LOG_DIR}/")


if __name__ == "__main__":
    import traceback

    run_task_matching_tests()
