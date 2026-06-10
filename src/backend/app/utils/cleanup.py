import asyncio
import copy
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.draft import TaskDraft
from app.models.schedule import MainScheduleSlot
from app.models.task import Task
from app.models.statistics import TaskStatistics, CategoryStatistics
from app.core.logging_config import setup_logging

logger = setup_logging()

TIME_SCORE_DECAY_FACTOR = 0.99
TIME_SCORE_MIN_THRESHOLD = 0.1


def _get_schedule_cutoff() -> datetime:
    """
    Get the cutoff datetime for schedule cleanup.
    Returns midnight (00:00) at the start of 3 days ago (local time).
    """
    now = datetime.now()
    three_days_ago = now.date() - timedelta(days=3)
    return datetime.combine(three_days_ago, datetime.min.time())


def sweep_drafts(db: Session):
    """
    Deletes all drafts older than 24 hours.
    """
    try:
        cutoff_time = datetime.now() - timedelta(hours=24)
        count = db.query(TaskDraft).filter(TaskDraft.created_at < cutoff_time).count()
        
        if count > 0:
            db.query(TaskDraft).filter(TaskDraft.created_at < cutoff_time).delete()
            db.commit()
            logger.info(f"Cleanup: Deleted {count} old drafts.")
        else:
            logger.info("Cleanup: No old drafts found.")
            
    except Exception as e:
        logger.error(f"Cleanup sweep failed: {e}")
        db.rollback()


def sweep_schedule(db: Session):
    """
    Deletes all main_schedule entries that ended before 3 days ago,
    along with their corresponding tasks.
    
    Keeps today's and the past 3 full days' entries.
    """
    try:
        cutoff = _get_schedule_cutoff()
        
        # Find all schedule slots that ended before cutoff
        old_slots = db.query(MainScheduleSlot).filter(
            MainScheduleSlot.end < cutoff
        ).all()
        
        if not old_slots:
            logger.info("Cleanup: No old schedule entries found.")
            return
        
        # Get unique task IDs
        task_ids = list(set(slot.task_id for slot in old_slots))
        
        # Delete tasks first (cascade handles main_schedule, provisional_schedule, etc.)
        deleted_tasks = db.query(Task).filter(Task.id.in_(task_ids)).delete(
            synchronize_session=False
        )
        
        db.commit()
        logger.info(
            f"Cleanup: Deleted {deleted_tasks} old tasks and {len(old_slots)} schedule entries "
            f"(ended before {cutoff.date()})."
        )
        
    except Exception as e:
        logger.error(f"Schedule cleanup failed: {e}")
        db.rollback()


def decay_time_scores(db: Session):
    """
    Multiply all time scores by 0.99 (decay over time).
    
    Applies to:
    - TaskStatistics.task_time_scores
    - CategoryStatistics.category_time_scores
    
    Values below 0.1 are set to 0.
    """
    try:
        all_task_stats = db.query(TaskStatistics).all()
        task_count = 0
        for stats in all_task_stats:
            try:
                time_scores = copy.deepcopy(stats.task_time_scores) or {}
                for slot in time_scores:
                    time_scores[slot] = round(time_scores[slot] * TIME_SCORE_DECAY_FACTOR, 5)
                    if abs(time_scores[slot]) < TIME_SCORE_MIN_THRESHOLD:
                        time_scores[slot] = 0
                stats.task_time_scores = time_scores
                task_count += 1
            except Exception as e:
                logger.error(f"Failed to decay task_stats {stats.id}: {e}")
        
        db.commit()
        logger.info(f"Decayed {task_count} task_statistics records")
        
        all_cat_stats = db.query(CategoryStatistics).all()
        cat_count = 0
        for stats in all_cat_stats:
            try:
                time_scores = copy.deepcopy(stats.category_time_scores) or {}
                for slot in time_scores:
                    time_scores[slot] = round(time_scores[slot] * TIME_SCORE_DECAY_FACTOR, 5)
                    if abs(time_scores[slot]) < TIME_SCORE_MIN_THRESHOLD:
                        time_scores[slot] = 0
                stats.category_time_scores = time_scores
                cat_count += 1
            except Exception as e:
                logger.error(f"Failed to decay category_stats {stats.id}: {e}")
        
        db.commit()
        logger.info(f"Decayed {cat_count} category_statistics records")
        
    except Exception as e:
        logger.error(f"Time score decay failed: {e}")
        db.rollback()


async def run_cleanup_loop():
    """
    Runs the cleanup job immediately on start, then every 24 hours.
    This is a background task that runs indefinitely.
    """
    logger.info("Background Cleanup Service Initialized.")
    
    while True:
        db = SessionLocal()
        try:
            sweep_drafts(db)
            sweep_schedule(db)
        except Exception as e:
            logger.error(f"Critical error in cleanup loop: {e}")
        finally:
            db.close()
            
        # Sleep for 24 hours (86400 seconds)
        logger.info("Cleanup sleeping for 24 hours...")
        await asyncio.sleep(86400)

        try:
            decay_time_scores(db)
        except Exception as e:
            logger.error(f"Critical error in cleanup loop: {e}")
        finally:
            db.close()
