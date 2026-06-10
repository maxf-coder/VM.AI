"""
Parser Service Integration Tests
10 tests for add mode, 12 tests for modify mode (including 2 error cases).
Logs to logs/parser_service_{test_type}_{DATE}.log

Run from src/backend directory:
    python tests/test_parser_service.py
"""

import sys
import os
import json
from datetime import datetime, timedelta
from uuid import UUID

# Add src/backend directory to path (same structure as test_enrichment.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.services.parser import parser_service
from app.schemas.task import TaskPayload
from app.schemas.nlp import NlpAddPayload

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs"
)


class TestLogger:
    """File-based logger for parser tests."""

    def __init__(self, test_type: str):
        self.test_type = test_type
        self.log_file = None
        self._open_log()

    def _open_log(self):
        """Open log file for this test type."""
        os.makedirs(LOG_DIR, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"parser_service_{self.test_type}_{date_str}.log"
        filepath = os.path.join(LOG_DIR, filename)
        self.log_file = open(filepath, "w", encoding="utf-8")
        print(f"Logging to: {filepath}")

    def write(self, content: str):
        """Write to log file."""
        self.log_file.write(content)

    def write_json(self, title: str, data: dict):
        """Write JSON data to log file."""
        self.write(f"\n--- {title} ---\n")
        self.write(json.dumps(serialize_for_json(data), indent=2) + "\n")

    def close(self):
        """Close log file."""
        if self.log_file:
            self.log_file.close()


def serialize_for_json(obj):
    """Serialize datetime, UUID, and other objects for JSON."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    return obj


def create_test_task() -> TaskPayload:
    """Create a valid TaskPayload for modify tests."""
    now = datetime.now()
    future_start = now + timedelta(days=7)
    future_deadline = now + timedelta(days=14)
    return TaskPayload(
        name="gym session",
        start=future_start,
        deadline=future_deadline,
        difficulty=0.5,
        duration=60,
        category=["fitness"],
        location="gym",
        importance=0.5,
        fixed_time=False,
        fixed_start=None,
    )


def test_add_mode():
    """Test parse_add with 10 test cases."""
    logger = TestLogger("add_mode")
    logger.write("=" * 70 + "\n")
    logger.write("PARSER SERVICE TEST: parse_add (10 test cases)\n")
    logger.write("=" * 70 + "\n")

    test_cases = [
        "gym session",
        "workout at gym for 1 hour",
        "study chemistry homework before Friday",
        "doctor appointment tomorrow",
        "easy yoga every monday",
        "urgent meeting at office",
        "buy groceries at supermarket",
        "pay rent on Friday",
        "code presentation for monday",
        "pick up kids from school",
    ]

    results = []

    for i, test_input in enumerate(test_cases, 1):
        logger.write(f"\n{'='*60}\n")
        logger.write(f"Test {i}: {test_input}\n")
        logger.write(f"{'='*60}\n")

        try:
            result = parser_service.parse_add(test_input)

            if result is None:
                logger.write("RESULT: None (ERROR)\n")
                results.append({"test_num": i, "input": test_input, "success": False, "error": "returned None"})
                continue

            logger.write(f"RESULT TYPE: {type(result).__name__}\n")

            # Verify NlpAddPayload
            if not isinstance(result, NlpAddPayload):
                logger.write(f"ERROR: Expected NlpAddPayload, got {type(result).__name__}\n")
                results.append({"test_num": i, "input": test_input, "success": False, "error": "wrong type"})
                continue

            # Log field values and types
            logger.write("\nFIELD VALUES AND TYPES:\n")
            result_dict = result.model_dump() if hasattr(result, "model_dump") else result.dict()

            # Check type conversion
            difficulty_val = result_dict.get("difficulty", {}).get("value")
            difficulty_type = type(difficulty_val).__name__

            duration_val = result_dict.get("duration", {}).get("value")
            duration_type = type(duration_val).__name__

            importance_val = result_dict.get("importance", {}).get("value")
            importance_type = type(importance_val).__name__

            logger.write(f"  difficulty: {difficulty_val} (type: {difficulty_type})\n")
            logger.write(f"  duration: {duration_val} (type: {duration_type})\n")
            logger.write(f"  importance: {importance_val} (type: {importance_type})\n")

            # Check type conversion
            type_errors = []
            if difficulty_type != "float":
                type_errors.append(f"difficulty is {difficulty_type}, expected float")
            if duration_type != "int":
                type_errors.append(f"duration is {duration_type}, expected int")
            if importance_type != "float":
                type_errors.append(f"importance is {importance_type}, expected float")

            # Check recurrent removed
            has_recurrent = "recurrent" in result_dict
            logger.write(f"  recurrent in output: {has_recurrent} (should be False)\n")

            if has_recurrent:
                type_errors.append("recurrent should be removed from output")

            logger.write_json("FULL OUTPUT", result_dict)

            if type_errors:
                results.append({"test_num": i, "input": test_input, "success": False, "error": "; ".join(type_errors)})
            else:
                results.append({"test_num": i, "input": test_input, "success": True})

        except Exception as e:
            logger.write(f"EXCEPTION: {str(e)}\n")
            results.append({"test_num": i, "input": test_input, "success": False, "error": str(e)})

    logger.write("\n" + "=" * 70 + "\n")
    logger.write("SUMMARY\n")
    logger.write("=" * 70 + "\n")

    passed = 0
    for r in results:
        status = "OK" if r["success"] else f"ERROR: {r.get('error', 'unknown')}"
        logger.write(f"Test {r['test_num']}: {r['input'][:30]:<30} - {status}\n")
        if r["success"]:
            passed += 1

    logger.write(f"\nPassed: {passed} / {len(results)}\n")
    logger.close()
    return results


def test_modify_mode():
    """Test parse_modify with 10 normal tests + 2 error cases = 12 tests."""
    logger = TestLogger("modify_mode")
    logger.write("=" * 70 + "\n")
    logger.write("PARSER SERVICE TEST: parse_modify (12 test cases)\n")
    logger.write("=" * 70 + "\n")

    test_cases = [
        ("make it urgent", "make it urgent"),
        ("make it harder", "make it harder"),
        ("change deadline to monday", "change deadline to monday"),
        ("set time to 3pm", "set time to 3pm"),
        ("make it optional", "make it optional"),
        ("push deadline to friday", "push deadline to friday"),
        ("change location to gym", "change location to gym"),
        ("increase duration to 2 hours", "increase duration to 2 hours"),
        ("make it easier", "make it easier"),
        ("cancel recurrence", "cancel recurrence"),
        ("empty string", ""),
        ("gibberish", "xyzabc123qwerty"),
    ]

    test_task = create_test_task()

    results = []

    for i, (change_desc, change_prompt) in enumerate(test_cases, 1):
        logger.write(f"\n{'='*60}\n")
        logger.write(f"Test {i}: {change_desc}\n")
        logger.write(f"{'='*60}\n")
        logger.write(f"Change prompt: {repr(change_prompt)}\n")

        try:
            result = parser_service.parse_modify(test_task, change_prompt)

            # Check error cases
            if i >= 11:
                # Error test cases - should return None
                if result is None:
                    logger.write("RESULT: None (expected for error case)\n")
                    results.append({"test_num": i, "change_desc": change_desc, "success": True})
                else:
                    logger.write(f"RESULT: {result} (expected None for error case)\n")
                    results.append({"test_num": i, "change_desc": change_desc, "success": False, "error": "should return None"})
                continue

            if result is None:
                logger.write("RESULT: None (ERROR)\n")
                results.append({"test_num": i, "change_desc": change_desc, "success": False, "error": "returned None"})
                continue

            # Verify it's a dict
            if not isinstance(result, dict):
                logger.write(f"ERROR: Expected dict, got {type(result).__name__}\n")
                results.append({"test_num": i, "change_desc": change_desc, "success": False, "error": "wrong type"})
                continue

            logger.write(f"RESULT TYPE: {type(result).__name__}\n")
            logger.write(f"RESULT KEYS: {list(result.keys())}\n")

            # Check type conversion for values
            type_errors = []
            for field, value in result.items():
                field_type = type(value).__name__

                if field in ("difficulty", "importance") and field_type != "float":
                    type_errors.append(f"{field} is {field_type}, expected float")
                if field == "duration" and field_type != "int":
                    type_errors.append(f"{field} is {field_type}, expected int")

            if type_errors:
                logger.write(f"TYPE ERRORS: {type_errors}\n")

            # Check invalid fields filtered
            invalid_fields = [k for k in result.keys() if k in ("recurrent", "recurrence_days")]
            if invalid_fields:
                logger.write(f"INVALID FIELDS (should be filtered): {invalid_fields}\n")
                type_errors.append("invalid fields not filtered")

            logger.write_json("RESULT", result)

            if type_errors:
                results.append({"test_num": i, "change_desc": change_desc, "success": False, "error": "; ".join(type_errors)})
            else:
                results.append({"test_num": i, "change_desc": change_desc, "success": True})

        except Exception as e:
            logger.write(f"EXCEPTION: {str(e)}\n")
            results.append({"test_num": i, "change_desc": change_desc, "success": False, "error": str(e)})

    logger.write("\n" + "=" * 70 + "\n")
    logger.write("SUMMARY\n")
    logger.write("=" * 70 + "\n")

    passed = 0
    for r in results:
        status = "OK" if r["success"] else f"ERROR: {r.get('error', 'unknown')}"
        logger.write(f"Test {r['test_num']:02d}: {r['change_desc'][:30]:<30} - {status}\n")
        if r["success"]:
            passed += 1

    logger.write(f"\nPassed: {passed} / {len(results)}\n")
    logger.close()
    return results


def main():
    print("\n" + "=" * 60)
    print("   PARSER SERVICE INTEGRATION TESTS")
    print("=" * 60)

    print("\n--- Test 1: PARSE_ADD ---")
    add_results = test_add_mode()
    print(f"Add mode completed: {len([r for r in add_results if r['success']])} / {len(add_results)} passed")

    print("\n--- Test 2: PARSE_MODIFY ---")
    modify_results = test_modify_mode()
    print(f"Modify mode completed: {len([r for r in modify_results if r['success']])} / {len(modify_results)} passed")

    print("\n" + "=" * 60)
    print("   TESTS COMPLETE")
    print("=" * 60)
    print(f"\nLogs saved to: {LOG_DIR}/parser_service_*.log")


if __name__ == "__main__":
    main()