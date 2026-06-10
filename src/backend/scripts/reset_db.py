"""
Reset all database tables - delete all content from existing tables.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models import (
    ScheduleChange,
    ProvisionalSlot,
    MainScheduleSlot,
    UnscheduledTask,
    TaskCategory,
    TaskStatisticsLocation,
    CategoryStatisticsLocation,
    Task,
    TaskDraft,
    Location,
    TaskStatistics,
    CategoryStatistics,
    Category,
)


def reset_db():
    """Delete all content from all database tables."""
    db = SessionLocal()
    try:
        print("Starting database reset...")

        db.query(ScheduleChange).delete()
        print("  Deleted schedule_changes")

        db.query(ProvisionalSlot).delete()
        print("  Deleted provisional_schedule")

        db.query(MainScheduleSlot).delete()
        print("  Deleted main_schedule")

        db.query(UnscheduledTask).delete()
        print("  Deleted unscheduled_tasks")

        db.query(TaskCategory).delete()
        print("  Deleted task_categories")

        db.query(TaskStatisticsLocation).delete()
        print("  Deleted task_statistics_locations")

        db.query(CategoryStatisticsLocation).delete()
        print("  Deleted category_statistics_locations")

        db.query(Task).delete()
        print("  Deleted tasks")

        db.query(TaskDraft).delete()
        print("  Deleted task_drafts")

        db.query(Location).delete()
        print("  Deleted locations")

        db.query(TaskStatistics).delete()
        print("  Deleted tasks_statistics")

        db.query(CategoryStatistics).delete()
        print("  Deleted category_statistics")

        db.query(Category).delete()
        print("  Deleted categories")

        db.commit()
        print("\nDatabase reset complete!")

    except Exception as e:
        db.rollback()
        print(f"Error during reset: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    reset_db()