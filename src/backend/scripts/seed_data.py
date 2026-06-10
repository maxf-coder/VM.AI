"""
Seed database with sample data for testing.

This script populates the database with realistic test data
that allows full functionality testing of the VM.AI backend.

Usage:
    python scripts/seed_data.py [--reset]

Insert Order (respecting FK dependencies):
    1. categories (5)
    2. locations (5)
    3. task_statistics (5) + category_statistics (5)
    4. task_statistics_locations (5) + category_statistics_locations (5)
    5. tasks (5)
    6. task_categories (5)
    7. unscheduled_tasks (5) + task_drafts (3)
"""
import sys
import os
import argparse
from datetime import datetime, timedelta
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models import (
    Category,
    Location,
    TaskStatistics,
    CategoryStatistics,
    TaskStatisticsLocation,
    CategoryStatisticsLocation,
    Task,
    TaskCategory,
    UnscheduledTask,
    TaskDraft,
)
from app.utils.task_saver import generate_default_time_scores, DEFAULT_TIME_ANCHORS


def _future_datetime(days_from_now: int = 0, hour: int = 6, minute: int = 0) -> datetime:
    """Generate a future datetime relative to now (naive format)."""
    now = datetime.now()
    future = now + timedelta(days=days_from_now)
    return future.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _check_existing_data(db) -> bool:
    """Check if any data already exists."""
    existing_category = db.query(Category).first()
    return existing_category is not None


def seed_data(db, reset: bool = False):
    """Main seeding function."""
    print("=" * 60)
    print("VM.AI Database Seeding Script")
    print("=" * 60)

    if _check_existing_data(db):
        if reset:
            print("\n[RESET MODE] Clearing existing data...")
            db.query(TaskDraft).delete()
            db.query(UnscheduledTask).delete()
            db.query(TaskCategory).delete()
            db.query(Task).delete()
            db.query(TaskStatisticsLocation).delete()
            db.query(CategoryStatisticsLocation).delete()
            db.query(CategoryStatistics).delete()
            db.query(TaskStatistics).delete()
            db.query(Location).delete()
            db.query(Category).delete()
            db.commit()
            print("[RESET COMPLETE]")
        else:
            print("\n[WARNING] Data already exists! Use --reset to clear first.")
            return

    # Load task matcher for real vectors
    print("\n[LOADING] Task matching model...")
    from app.services.task_matcher import task_matcher
    task_matcher._load_model()
    print("[LOAD COMPLETE]")

    # ============================================================================
    # PHASE 1: Independent tables (no FK dependencies)
    # ============================================================================
    print("\n[PHASE 1] Creating categories and locations...")

    # Categories
    category_data = [
        "study",
        "fitness",
        "work",
        "personal",
        "hobby",
    ]
    categories = {}
    for name in category_data:
        cat = Category(name=name)
        db.add(cat)
        db.flush()
        categories[name] = cat
    print(f"  Created {len(categories)} categories")

    # Locations
    location_data = [
        "home",
        "office",
        "library",
        "gym",
        "store",
    ]
    locations = {}
    for name in location_data:
        loc = Location(name=name)
        db.add(loc)
        db.flush()
        locations[name] = loc
    print(f"  Created {len(locations)} locations")

    # ============================================================================
    # PHASE 2: Statistics tables
    # ============================================================================
    print("\n[PHASE 2] Creating task and category statistics...")

    # Task Statistics (5 records with real vectors)
    task_stats_data = [
        {
            "name": "Gym session",
            "avg_difficulty": 0.65,
            "avg_duration": {"0.5": {"avg": 60.0, "count": 1}, "1.0": {"avg": 90.0, "count": 2}},
            "records": 3,
            "completed_count": 2,
            "uncompleted_count": 1,
        },
        {
            "name": "Study session",
            "avg_difficulty": 0.70,
            "avg_duration": {"0.5": {"avg": 120.0, "count": 2}},
            "records": 2,
            "completed_count": 1,
            "uncompleted_count": 1,
        },
        {
            "name": "Team meeting",
            "avg_difficulty": 0.60,
            "avg_duration": {"0.5": {"avg": 60.0, "count": 1}},
            "records": 1,
            "completed_count": 0,
            "uncompleted_count": 1,
        },
        {
            "name": "Buy groceries",
            "avg_difficulty": 0.30,
            "avg_duration": {"0.0": {"avg": 45.0, "count": 1}},
            "records": 1,
            "completed_count": 0,
            "uncompleted_count": 1,
        },
        {
            "name": "Project presentation",
            "avg_difficulty": 0.75,
            "avg_duration": {"1.0": {"avg": 480.0, "count": 1}},
            "records": 1,
            "completed_count": 0,
            "uncompleted_count": 1,
        },
    ]
    task_statistics = {}
    for data in task_stats_data:
        vector = task_matcher.model.encode(data["name"], normalize_embeddings=True).tolist()
        stats = TaskStatistics(
            task_name=data["name"],
            task_name_vector=vector,
            avg_duration=data["avg_duration"],
            avg_duration_delta={},
            avg_difficulty=data["avg_difficulty"],
            avg_difficulty_delta=0.0,
            completed_count=data["completed_count"],
            uncompleted_count=data["uncompleted_count"],
            records=data["records"],
            task_time_scores=generate_default_time_scores(DEFAULT_TIME_ANCHORS),
        )
        db.add(stats)
        db.flush()
        task_statistics[data["name"]] = stats
    print(f"  Created {len(task_statistics)} task_statistics records")

    # Category Statistics (5 records)
    cat_stats_data = [
        {"category": "study", "avg_difficulty": 0.70, "avg_duration": {"0.5": {"avg": 90.0, "count": 2}}, "records": 2, "completed": 1, "uncompleted": 1},
        {"category": "fitness", "avg_difficulty": 0.65, "avg_duration": {"0.5": {"avg": 60.0, "count": 1}, "1.0": {"avg": 75.0, "count": 2}}, "records": 3, "completed": 2, "uncompleted": 1},
        {"category": "work", "avg_difficulty": 0.68, "avg_duration": {"0.5": {"avg": 270.0, "count": 2}}, "records": 2, "completed": 0, "uncompleted": 2},
        {"category": "personal", "avg_difficulty": 0.30, "avg_duration": {"0.0": {"avg": 45.0, "count": 1}}, "records": 1, "completed": 0, "uncompleted": 1},
        {"category": "hobby", "avg_difficulty": 0.40, "avg_duration": {"0.5": {"avg": 120.0, "count": 1}}, "records": 1, "completed": 0, "uncompleted": 1},
    ]
    category_statistics = {}
    for data in cat_stats_data:
        cat_id = categories[data["category"]].id
        stats = CategoryStatistics(
            category_id=cat_id,
            avg_duration=data["avg_duration"],
            avg_duration_delta={},
            avg_difficulty=data["avg_difficulty"],
            avg_difficulty_delta=0.0,
            completed_count=data["completed"],
            uncompleted_count=data["uncompleted"],
            records=data["records"],
            category_time_scores=generate_default_time_scores(DEFAULT_TIME_ANCHORS),
        )
        db.add(stats)
        db.flush()
        category_statistics[data["category"]] = stats
    print(f"  Created {len(category_statistics)} category_statistics records")

    # ============================================================================
    # PHASE 3: Location statistics (junction tables)
    # ============================================================================
    print("\n[PHASE 3] Creating location statistics...")

    # Task Statistics Locations
    task_stats_loc_data = [
        {"task": "Gym session", "location": "gym", "count": 3},
        {"task": "Study session", "location": "library", "count": 2},
        {"task": "Team meeting", "location": "office", "count": 1},
        {"task": "Buy groceries", "location": "store", "count": 1},
        {"task": "Project presentation", "location": "office", "count": 1},
    ]
    for data in task_stats_loc_data:
        rel = TaskStatisticsLocation(
            statistics_id=task_statistics[data["task"]].id,
            location_id=locations[data["location"]].id,
            count=data["count"],
        )
        db.add(rel)
    print(f"  Created {len(task_stats_loc_data)} task_statistics_locations records")

    # Category Statistics Locations
    cat_stats_loc_data = [
        {"category": "study", "location": "library", "count": 2},
        {"category": "fitness", "location": "gym", "count": 3},
        {"category": "work", "location": "office", "count": 2},
        {"category": "personal", "location": "store", "count": 1},
        {"category": "hobby", "location": "home", "count": 1},
    ]
    for data in cat_stats_loc_data:
        rel = CategoryStatisticsLocation(
            statistics_id=category_statistics[data["category"]].id,
            location_id=locations[data["location"]].id,
            count=data["count"],
        )
        db.add(rel)
    print(f"  Created {len(cat_stats_loc_data)} category_statistics_locations records")

    db.flush()

    # ============================================================================
    # PHASE 4: Tasks (depends on task_statistics and locations)
    # ============================================================================
    print("\n[PHASE 4] Creating tasks...")

    task_data = [
        {
            "name": "Gym session",
            "task_statistics": "Gym session",
            "location": "gym",
            "difficulty": 0.70,
            "duration": 75,
            "importance": 0.60,
            "category": "fitness",
            "start_days": 1,
            "start_hour": 6,
            "deadline_days": 1,
            "deadline_hour": 23,
            "deadline_minute": 59,
        },
        {
            "name": "Study session",
            "task_statistics": "Study session",
            "location": "library",
            "difficulty": 0.65,
            "duration": 120,
            "importance": 0.75,
            "category": "study",
            "start_days": 2,
            "start_hour": 9,
            "deadline_days": 2,
            "deadline_hour": 21,
            "deadline_minute": 0,
        },
        {
            "name": "Team meeting",
            "task_statistics": "Team meeting",
            "location": "office",
            "difficulty": 0.55,
            "duration": 60,
            "importance": 0.80,
            "category": "work",
            "start_days": 3,
            "start_hour": 14,
            "deadline_days": 3,
            "deadline_hour": 15,
            "deadline_minute": 0,
        },
        {
            "name": "Buy groceries",
            "task_statistics": "Buy groceries",
            "location": "store",
            "difficulty": 0.25,
            "duration": 45,
            "importance": 0.50,
            "category": "personal",
            "start_days": 1,
            "start_hour": 17,
            "deadline_days": 1,
            "deadline_hour": 18,
            "deadline_minute": 0,
        },
        {
            "name": "Project presentation",
            "task_statistics": "Project presentation",
            "location": "office",
            "difficulty": 0.80,
            "duration": 480,
            "importance": 0.90,
            "category": "work",
            "start_days": 5,
            "start_hour": 9,
            "deadline_days": 5,
            "deadline_hour": 17,
            "deadline_minute": 0,
        },
    ]

    tasks = {}
    for data in task_data:
        urgency = 0.5
        value = (data["importance"] * 0.4) + (urgency * 0.4) + (data["difficulty"] * 0.2)

        task = Task(
            task_statistics_id=task_statistics[data["task_statistics"]].id,
            associated_task_statistics_id=None,
            name=data["name"],
            start=_future_datetime(data["start_days"], data["start_hour"]),
            deadline=_future_datetime(data["deadline_days"], data["deadline_hour"], data["deadline_minute"]),
            difficulty=data["difficulty"],
            duration=data["duration"],
            location_id=locations[data["location"]].id,
            importance=data["importance"],
            urgency=urgency,
            value=round(value, 2),
            fixed_time=False,
            fixed_start=None,
            rated=False,
        )
        db.add(task)
        db.flush()
        tasks[data["name"]] = task

    print(f"  Created {len(tasks)} tasks")

    # ============================================================================
    # PHASE 5: Task Categories (junction table)
    # ============================================================================
    print("\n[PHASE 5] Creating task categories...")

    task_category_data = [
        {"task": "Gym session", "category": "fitness", "priority": 0},
        {"task": "Study session", "category": "study", "priority": 0},
        {"task": "Team meeting", "category": "work", "priority": 0},
        {"task": "Buy groceries", "category": "personal", "priority": 0},
        {"task": "Project presentation", "category": "work", "priority": 0},
    ]

    for data in task_category_data:
        tc = TaskCategory(
            task_id=tasks[data["task"]].id,
            category_id=categories[data["category"]].id,
            priority=data["priority"],
        )
        db.add(tc)
    print(f"  Created {len(task_category_data)} task_categories records")

    # ============================================================================
    # PHASE 6: Workflow tables (unscheduled tasks)
    # ============================================================================
    print("\n[PHASE 6] Creating unscheduled tasks...")

    for task_name, task in tasks.items():
        unscheduled = UnscheduledTask(task_id=task.id)
        db.add(unscheduled)
    print(f"  Created {len(tasks)} unscheduled_tasks records")

    # ============================================================================
    # PHASE 7: Task drafts (sample drafts with JSONB content)
    # ============================================================================
    print("\n[PHASE 7] Creating task drafts...")

    draft_data = [
        {
            "task": {
                "name": "Morning run",
                "start": _future_datetime(1, 6).isoformat(),
                "deadline": _future_datetime(1, 7).isoformat(),
                "difficulty": 0.5,
                "duration": 60,
                "category": ["fitness"],
                "location": "park",
                "importance": 0.6,
                "fixed_time": False,
                "fixed_start": None,
            },
            "match_result": {
                "associated_id": str(task_statistics["Gym session"].id),
                "association_status": "similar",
                "name_vector": task_matcher.model.encode("Morning run", normalize_embeddings=True).tolist(),
            },
        },
        {
            "task": {
                "name": "Read chapter",
                "start": _future_datetime(2, 20).isoformat(),
                "deadline": _future_datetime(2, 22).isoformat(),
                "difficulty": 0.3,
                "duration": 120,
                "category": ["study", "hobby"],
                "location": "home",
                "importance": 0.4,
                "fixed_time": False,
                "fixed_start": None,
            },
            "match_result": {
                "associated_id": str(task_statistics["Study session"].id),
                "association_status": "similar",
                "name_vector": task_matcher.model.encode("Read chapter", normalize_embeddings=True).tolist(),
            },
        },
        {
            "task": {
                "name": "Grocery shopping",
                "start": _future_datetime(1, 18).isoformat(),
                "deadline": _future_datetime(1, 19).isoformat(),
                "difficulty": 0.2,
                "duration": 60,
                "category": ["personal"],
                "location": "supermarket",
                "importance": 0.5,
                "fixed_time": False,
                "fixed_start": None,
            },
            "match_result": {
                "associated_id": str(task_statistics["Buy groceries"].id),
                "association_status": "similar",
                "name_vector": task_matcher.model.encode("Grocery shopping", normalize_embeddings=True).tolist(),
            },
        },
    ]

    for data in draft_data:
        draft = TaskDraft(id=uuid4(), content=data)
        db.add(draft)
    print(f"  Created {len(draft_data)} task_drafts records")

    # ============================================================================
    # COMMIT
    # ============================================================================
    print("\n[COMMIT] Saving all data to database...")
    db.commit()

    print("\n" + "=" * 60)
    print("SEEDING COMPLETE!")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - {len(categories)} categories")
    print(f"  - {len(locations)} locations")
    print(f"  - {len(task_statistics)} task_statistics")
    print(f"  - {len(category_statistics)} category_statistics")
    print(f"  - {len(task_stats_loc_data)} task_statistics_locations")
    print(f"  - {len(cat_stats_loc_data)} category_statistics_locations")
    print(f"  - {len(tasks)} tasks")
    print(f"  - {len(task_category_data)} task_categories")
    print(f"  - {len(tasks)} unscheduled_tasks")
    print(f"  - {len(draft_data)} task_drafts")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Seed database with sample data")
    parser.add_argument("--reset", action="store_true", help="Reset (clear) existing data before seeding")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        seed_data(db, reset=args.reset)
    except Exception as e:
        db.rollback()
        print(f"\n[ERROR] Seeding failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
