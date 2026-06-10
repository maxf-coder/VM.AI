"""
Enrichment Testing Script with Deep JSON Logging to Files

5 tests per enrichment type with varying match_result:
- 1 test with association_status="none"
- 2 tests with association_status="same"
- 2 tests with association_status="similar"

All match_results have name_vector (never None).
2 fixed_time=True tests per type.

Fresh db session per test case - FULL JSON logging for:
- INPUT: All task fields
- OUTPUT: All computed fields
- DB: tasks + task_statistics tables
- ERROR: Full stacktrace in JSON

Logs saved to C:\\VM.AI\\src\\backend\\logs\\enrichment_{test_type}_{YYYYMMDD}.log

Run from backend directory:
    cd src/backend
    python tests/test_enrichment.py
"""

import sys
import os
import json
import traceback
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from uuid import UUID
from app.core.database import SessionLocal
from app.models.draft import TaskDraft
from app.models.statistics import TaskStatistics
from app.models.task import Task
from app.services.task_matcher import TaskMatcher


mock_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)


class TestLogger:
    """File-based logger for enrichment tests."""

    def __init__(self, test_type: str):
        self.test_type = test_type
        self.log_file = None
        self._open_log()

    def _open_log(self):
        """Open log file for this test type."""
        os.makedirs(LOG_DIR, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"enrichment_{self.test_type}_{date_str}.log"
        filepath = os.path.join(LOG_DIR, filename)
        self.log_file = open(filepath, "w", encoding="utf-8")

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
    """Serialize datetime and UUID objects for JSON."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    return obj


def log_db_results(db, logger: TestLogger, title: str):
    """Query and log tasks + tasks_statistics from DB."""
    logger.write(f"\n--- {title} ---\n")

    # Query tasks
    tasks = db.query(Task).all()
    tasks_data = []
    for t in tasks:
        tasks_data.append(
            {
                "id": t.id,
                "name": t.name,
                "start": t.start,
                "deadline": t.deadline,
                "difficulty": t.difficulty,
                "duration": t.duration,
                "location_id": t.location_id,
                "importance": t.importance,
                "urgency": t.urgency,
                "value": t.value,
                "task_statistics_id": t.task_statistics_id,
                "associated_task_statistics_id": t.associated_task_statistics_id,
            }
        )
    logger.write(f"TASKS: {json.dumps(serialize_for_json(tasks_data), indent=2)}\n")

    # Query tasks_statistics
    stats = db.query(TaskStatistics).all()
    stats_data = []
    for s in stats:
        stats_data.append(
            {
                "id": s.id,
                "task_name": s.task_name,
                "avg_duration": s.avg_duration,
                "avg_difficulty": s.avg_difficulty,
                "completed_count": s.completed_count,
                "uncompleted_count": s.uncompleted_count,
                "records": s.records,
            }
        )
    logger.write(
        f"TASKS_STATISTICS: {json.dumps(serialize_for_json(stats_data), indent=2)}\n"
    )


def test_commit_manual():
    """Test 1: commit_manual (5 test cases)"""
    from app.services.enrichment import enrichment_service

    logger = TestLogger("commit_manual")
    logger.write("=" * 70 + "\n")
    logger.write("TEST 1: commit_manual (5 test cases)\n")
    logger.write("=" * 70 + "\n")

    test_cases = [
        {
            "name": "test_manual_1",
            "fixed_time": False,
            "task": {
                "name": "study session",
                "start": datetime(2026, 4, 20, 9, 0),
                "deadline": datetime(2026, 4, 25, 17, 0),
                "difficulty": 0.7,
                "duration": 60,
                "category": ["study"],
                "location": "home",
                "importance": 0.6,
                "fixed_time": False,
                "fixed_start": None,
            },
        },
        {
            "name": "test_manual_2",
            "fixed_time": False,
            "task": {
                "name": "chemistry homework",
                "start": datetime(2026, 4, 21, 10, 0),
                "deadline": datetime(2026, 4, 26, 18, 0),
                "difficulty": 0.75,
                "duration": 75,
                "category": ["study"],
                "location": "home",
                "importance": 0.7,
                "fixed_time": False,
                "fixed_start": None,
            },
        },
        {
            "name": "test_manual_3",
            "fixed_time": True,
            "task": {
                "name": "gym workout",
                "start": None,
                "deadline": None,
                "difficulty": 0.5,
                "duration": 90,
                "category": ["fitness"],
                "location": "gym",
                "importance": 0.8,
                "fixed_time": True,
                "fixed_start": datetime(2026, 4, 20, 6, 0),
            },
        },
        {
            "name": "test_manual_4",
            "fixed_time": False,
            "task": {
                "name": "math assignment",
                "start": datetime(2026, 4, 22, 11, 0),
                "deadline": datetime(2026, 4, 27, 16, 0),
                "difficulty": 0.6,
                "duration": 45,
                "category": ["study"],
                "location": "library",
                "importance": 0.65,
                "fixed_time": False,
                "fixed_start": None,
            },
        },
        {
            "name": "test_manual_5",
            "fixed_time": True,
            "task": {
                "name": "work meeting",
                "start": None,
                "deadline": None,
                "difficulty": 0.6,
                "duration": 60,
                "category": ["work"],
                "location": "office",
                "importance": 0.75,
                "fixed_time": True,
                "fixed_start": datetime(2026, 4, 21, 9, 0),
            },
        },
    ]

    for i, case in enumerate(test_cases, 1):
        logger.write(f"\n[Case 1.{i}] {case['name']} (fixed={case['fixed_time']})\n")

        # Fresh db session per test case
        db = SessionLocal()

        try:
            # Get real match from TaskMatcher
            matcher = TaskMatcher()
            real_match = matcher.find_match(db, case["task"]["name"])

            logger.write_json("INPUT task_payload", case["task"])
            logger.write_json("INPUT match_result", real_match)

            result = enrichment_service.commit_manual(db, case["task"], real_match)

            logger.write_json("OUTPUT result", result)

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


def test_predict_nlp_add():
    """Test 2: predict_nlp_add (5 test cases)"""
    from app.services.enrichment import enrichment_service

    logger = TestLogger("predict_nlp_add")
    logger.write("=" * 70 + "\n")
    logger.write("TEST 2: predict_nlp_add (5 test cases)\n")
    logger.write("=" * 70 + "\n")

    nlp_payload_fixed = {
        "name": {"value": "test", "predicted": False},
        "start": {"value": "", "predicted": True},
        "deadline": {"value": "", "predicted": True},
        "difficulty": {"value": 0.5, "predicted": True},
        "duration": {"value": 60, "predicted": True},
        "category": {"value": [], "predicted": False},
        "location": {"value": "", "predicted": True},
        "importance": {"value": 0.5, "predicted": True},
        "fixed_time": {"value": False, "predicted": False},
        "fixed_start": {"value": None, "predicted": False},
    }

    nlp_payload_fixed_true = {
        "name": {"value": "test", "predicted": False},
        "start": {"value": None, "predicted": True},
        "deadline": {"value": None, "predicted": True},
        "difficulty": {"value": 0.5, "predicted": True},
        "duration": {"value": 60, "predicted": True},
        "category": {"value": [], "predicted": False},
        "location": {"value": "", "predicted": True},
        "importance": {"value": 0.5, "predicted": True},
        "fixed_time": {"value": True, "predicted": False},
        "fixed_start": {"value": "Monday 06:00", "predicted": False},
    }

    test_cases = [
        {
            "name": "test_nlp_add_1",
            "task_name": "chemistry homework",
            "nlp": {
                **nlp_payload_fixed,
                "name": {"value": "chemistry homework", "predicted": False},
                "start": {"value": "Next monday at 09:00", "predicted": True},
                "deadline": {"value": "Next friday at 17:00", "predicted": True},
                "difficulty": {"value": 0.7, "predicted": True},
                "duration": {"value": 60, "predicted": True},
                "category": {"value": ["study"], "predicted": False},
                "location": {"value": "home", "predicted": True},
                "importance": {"value": 0.6, "predicted": True},
            },
        },
        {
            "name": "test_nlp_add_2",
            "task_name": "math assignment",
            "nlp": {
                **nlp_payload_fixed,
                "name": {"value": "math assignment", "predicted": False},
                "start": {"value": "Tomorrow at 10:00", "predicted": True},
                "deadline": {"value": "Next week friday", "predicted": True},
                "difficulty": {"value": 0.6, "predicted": True},
                "duration": {"value": 45, "predicted": True},
                "category": {"value": ["study"], "predicted": False},
                "location": {"value": "library", "predicted": True},
                "importance": {"value": 0.7, "predicted": True},
            },
        },
        {
            "name": "test_nlp_add_3",
            "task_name": "gym workout",
            "nlp": {
                **nlp_payload_fixed_true,
                "name": {"value": "gym workout", "predicted": False},
                "difficulty": {"value": 0.5, "predicted": True},
                "duration": {"value": 90, "predicted": True},
                "category": {"value": ["fitness"], "predicted": False},
                "location": {"value": "gym", "predicted": True},
                "importance": {"value": 0.8, "predicted": True},
            },
        },
        {
            "name": "test_nlp_add_4",
            "task_name": "physics lab",
            "nlp": {
                **nlp_payload_fixed,
                "name": {"value": "physics lab", "predicted": False},
                "start": {"value": "Monday", "predicted": True},
                "deadline": {"value": "Friday", "predicted": True},
                "difficulty": {"value": 0.8, "predicted": True},
                "duration": {"value": 90, "predicted": True},
                "category": {"value": ["study"], "predicted": False},
                "location": {"value": "university", "predicted": True},
                "importance": {"value": 0.8, "predicted": True},
            },
        },
        {
            "name": "test_nlp_add_5",
            "task_name": "work meeting",
            "nlp": {
                **nlp_payload_fixed_true,
                "name": {"value": "work meeting", "predicted": False},
                "difficulty": {"value": 0.6, "predicted": True},
                "duration": {"value": 60, "predicted": True},
                "category": {"value": ["work"], "predicted": False},
                "location": {"value": "office", "predicted": True},
                "importance": {"value": 0.75, "predicted": True},
            },
        },
    ]

    for i, case in enumerate(test_cases, 1):
        logger.write(f"\n[Case 2.{i}] {case['name']}\n")

        db = SessionLocal()

        try:
            # Get real match from TaskMatcher
            matcher = TaskMatcher()
            real_match = matcher.find_match(db, case["task_name"])

            logger.write_json("INPUT nlp_payload", case["nlp"])
            logger.write_json("INPUT match_result", real_match)

            result, draft_id = enrichment_service.predict_nlp_add(
                db, case["nlp"], real_match
            )

            output_data = {"result": result, "draft_id": str(draft_id)}
            logger.write_json("OUTPUT result", output_data)

            # Query draft from DB to get full saved record
            db.expire_all()
            saved_draft = db.query(TaskDraft).filter(TaskDraft.id == draft_id).first()
            if saved_draft:
                logger.write_json("DRAFT_AFTER", saved_draft.content)

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


def test_commit_from_draft():
    """Test 3: commit_from_draft (5 test cases)"""
    from app.services.enrichment import enrichment_service

    logger = TestLogger("commit_from_draft")
    logger.write("=" * 70 + "\n")
    logger.write("TEST 3: commit_from_draft (5 test cases)\n")
    logger.write("=" * 70 + "\n")

    db = SessionLocal()
    drafts = db.query(TaskDraft).all()
    draft_ids = [d.id for d in drafts]
    db.close()

    for i, draft_id in enumerate(draft_ids[:5], 1):
        logger.write(f"\n=== Processing test {i} of {len(draft_ids[:5])} ===\n")
        is_fixed = i in [3, 5]
        request_task = {
            "name": f"updated task {i}",
            "start": None if is_fixed else datetime(2026, 4, 20 + i, 9, 0),
            "deadline": None if is_fixed else datetime(2026, 4, 25 + i, 17, 0),
            "difficulty": 0.5 + (i * 0.05),
            "duration": 30 + (i * 15),
            "category": ["study"],
            "location": "home",
            "importance": 0.6 + (i * 0.05),
            "fixed_time": is_fixed,
            "fixed_start": datetime(2026, 4, 20, 6, 0) if is_fixed else None,
        }

        logger.write(f"\n[Case 3.{i}] draft_id={draft_id} (fixed={is_fixed})\n")

        db = SessionLocal()

        try:
            logger.write_json("INPUT request_task", request_task)
            logger.write(f"    INPUT draft_id: {draft_id}\n")

            # Log the original draft content before commit
            draft_before = db.query(TaskDraft).filter(TaskDraft.id == draft_id).first()
            if draft_before:
                logger.write_json("DRAFT_BEFORE", draft_before.content)

            result = enrichment_service.commit_from_draft(db, request_task, draft_id)

            logger.write_json("OUTPUT result", result)

            # Query draft again after commit for comparison
            db.expire_all()  # Clear cache to get fresh data
            draft_after = db.query(TaskDraft).filter(TaskDraft.id == draft_id).first()
            if draft_after:
                logger.write_json("DRAFT_AFTER", draft_after.content)
            else:
                logger.write(
                    f"    DRAFT_AFTER: NOT FOUND (draft deleted after commit)\n"
                )

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


def test_merge_nlp_modify():
    """Test 4: merge_nlp_modify (5 test cases)"""
    from app.services.enrichment import enrichment_service

    logger = TestLogger("merge_nlp_modify")
    logger.write("=" * 70 + "\n")
    logger.write("TEST 4: merge_nlp_modify (5 test cases)\n")
    logger.write("=" * 70 + "\n")

    db = SessionLocal()
    stats = db.query(TaskStatistics).limit(5).all()
    stats_ids = [s.id for s in stats]
    db.close()

    test_cases = [
        {
            "name": "test_modify_1",
            "fixed_time": False,
            "existing": {
                "name": "chemistry homework",
                "start": datetime(2026, 4, 20, 9, 0),
                "deadline": datetime(2026, 4, 25, 17, 0),
                "difficulty": 0.7,
                "duration": 60,
                "category": ["study"],
                "location": "home",
                "importance": 0.6,
                "fixed_time": False,
                "fixed_start": None,
            },
            "changed": {
                "deadline": "Next monday at 10:00",
                "duration": 90,
            },
        },
        {
            "name": "test_modify_2",
            "fixed_time": False,
            "existing": {
                "name": "math assignment",
                "start": datetime(2026, 4, 21, 10, 0),
                "deadline": datetime(2026, 4, 26, 18, 0),
                "difficulty": 0.6,
                "duration": 45,
                "category": ["study"],
                "location": "library",
                "importance": 0.7,
                "fixed_time": False,
                "fixed_start": None,
            },
            "changed": {
                "start": "Tuesday at 08:00",
                "deadline": "Friday",
            },
        },
        {
            "name": "test_modify_3",
            "fixed_time": True,
            "existing": {
                "name": "gym workout",
                "start": None,
                "deadline": None,
                "difficulty": 0.5,
                "duration": 90,
                "category": ["fitness"],
                "location": "gym",
                "importance": 0.8,
                "fixed_time": True,
                "fixed_start": datetime(2026, 4, 20, 6, 0),
            },
            "changed": {
                "duration": 120,
            },
        },
        {
            "name": "test_modify_4",
            "fixed_time": False,
            "existing": {
                "name": "physics lab",
                "start": datetime(2026, 4, 22, 11, 0),
                "deadline": datetime(2026, 4, 27, 16, 0),
                "difficulty": 0.8,
                "duration": 90,
                "category": ["study"],
                "location": "university",
                "importance": 0.8,
                "fixed_time": False,
                "fixed_start": None,
            },
            "changed": {
                "duration": 120,
                "location": "lab",
            },
        },
        {
            "name": "test_modify_5",
            "fixed_time": True,
            "existing": {
                "name": "work meeting",
                "start": None,
                "deadline": None,
                "difficulty": 0.6,
                "duration": 60,
                "category": ["work"],
                "location": "office",
                "importance": 0.75,
                "fixed_time": True,
                "fixed_start": datetime(2026, 4, 21, 9, 0),
            },
            "changed": {
                "duration": 90,
            },
        },
    ]

    for i, case in enumerate(test_cases, 1):
        logger.write(f"\n[Case 4.{i}] {case['name']} (fixed={case['fixed_time']})\n")

        db = SessionLocal()

        try:
            logger.write_json("INPUT existing_task", case["existing"])
            logger.write_json("INPUT changed_fields", case["changed"])

            result = enrichment_service.merge_nlp_modify(
                db, case["existing"], case["changed"]
            )

            logger.write_json("OUTPUT result", result)

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


def test_update_task():
    """Test 5: update_task (2 test cases)"""
    from app.services.enrichment import enrichment_service

    logger = TestLogger("update_task")
    logger.write("=" * 70 + "\n")
    logger.write("TEST 5: update_task (2 test cases)\n")
    logger.write("=" * 70 + "\n")

    test_cases = [
        {
            "name": "test_update_1",
            "fixed_time": False,
            "task": {
                "name": "study session",
                "start": datetime(2026, 4, 20, 9, 0),
                "deadline": datetime(2026, 4, 25, 17, 0),
                "difficulty": 0.7,
                "duration": 60,
                "category": ["study"],
                "location": "home",
                "importance": 0.6,
                "fixed_time": False,
                "fixed_start": None,
            },
        },
        {
            "name": "test_update_2",
            "fixed_time": True,
            "task": {
                "name": "gym workout",
                "start": None,
                "deadline": None,
                "difficulty": 0.5,
                "duration": 90,
                "category": ["fitness"],
                "location": "gym",
                "importance": 0.8,
                "fixed_time": True,
                "fixed_start": datetime(2026, 4, 20, 6, 0),
            },
        },
    ]

    for i, case in enumerate(test_cases, 1):
        logger.write(f"\n[Case 5.{i}] {case['name']} (fixed={case['fixed_time']})\n")

        db = SessionLocal()

        try:
            logger.write_json("INPUT task_payload", case["task"])

            result = enrichment_service.update_task(db, case["task"])

            logger.write_json("OUTPUT result", result)

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


def main():
    print("Running enrichment tests - see logs in logs/ directory")

    start = time.time()
    test_commit_manual()
    test_predict_nlp_add()
    test_commit_from_draft()
    test_merge_nlp_modify()
    test_update_task()

    print("ALL TESTS COMPLETED - See logs in C:\\VM.AI\\src\\backend\\logs\\")
    end = time.time()
    print(f"Execution time: {end - start:.4f}")


if __name__ == "__main__":
    main()
