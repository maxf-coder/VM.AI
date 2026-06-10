import copy
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Task, TaskStatistics, CategoryStatistics, TaskCategory
from app.models.statistics import (
    TaskStatisticsLocation,
    CategoryStatisticsLocation,
)
from app.core.logging_config import setup_logging

logger = setup_logging()

RECORDS_NR_TRACK = 30

TIME_SCORE_CLAMP = (-10.0, 10.0)
TIME_SCORE_STEP = 0.25


class StatsRecorder:
    """
    Service for recording task statistics after commit.
    
    Updates:
    - TaskStatistics (avg_difficulty, avg_duration, records)
    - TaskStatisticsLocation (count per location)
    - CategoryStatistics for each category
    - CategoryStatisticsLocation (count per location)
    """

    RECORDS_NR_TRACK = RECORDS_NR_TRACK

    def update_stats_after_commit(self, db: Session, task_uuid: UUID) -> bool:
        """
        Update statistics after a task is committed.
        
        Args:
            db: Database session
            task_uuid: UUID of the committed task
            
        Returns:
            bool: True if successful, False on error
        """
        try:
            # Fetch task from DB
            task = db.query(Task).filter(Task.id == task_uuid).first()
            if not task:
                logger.error(f"Task not found: {task_uuid}")
                return False

            # Update TaskStatistics
            self._update_task_statistics(db, task)

            # Update TaskStatisticsLocation
            self._update_task_statistics_location(db, task)

            # Update CategoryStatistics for each category
            self._update_category_statistics(db, task)

            # Update CategoryStatisticsLocation
            self._update_category_statistics_location(db, task)

            db.commit()
            logger.info(f"Stats updated for task: {task.name}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update stats for task {task_uuid}: {e}")
            return False

    def _update_task_statistics(self, db: Session, task: Task) -> None:
        """Update TaskStatistics with new task data."""
        stats = db.query(TaskStatistics).filter(
            TaskStatistics.id == task.task_statistics_id
        ).first()

        if not stats:
            logger.warning(f"TaskStatistics not found for task: {task.name}")
            return

        # Get current values
        current_records = stats.records or 0
        current_difficulty = stats.avg_difficulty or 0.0

        # EMA for avg_difficulty: smooth transition from cumulative to rolling window
        alpha = 1.0 / min(current_records + 1, self.RECORDS_NR_TRACK)
        new_difficulty = round((1 - alpha) * current_difficulty + alpha * task.difficulty, 2)
        stats.avg_difficulty = new_difficulty

        # Update duration bucket
        bucket = self._get_duration_bucket(task.difficulty)
        avg_duration = copy.deepcopy(stats.avg_duration) if stats.avg_duration else {}

        if bucket not in avg_duration:
            avg_duration[bucket] = {"count": 0, "avg": 0}

        bucket_count = avg_duration[bucket].get("count", 0)
        bucket_avg = avg_duration[bucket].get("avg", 0)

        # EMA for duration bucket
        bucket_alpha = 1.0 / min(bucket_count + 1, self.RECORDS_NR_TRACK)
        new_bucket_avg = round((1 - bucket_alpha) * bucket_avg + bucket_alpha * task.duration, 2)
        avg_duration[bucket] = {
            "count": min(bucket_count + 1, self.RECORDS_NR_TRACK),
            "avg": int(new_bucket_avg)
        }
        stats.avg_duration = avg_duration

        # Increment records (capped)
        stats.records = min(current_records + 1, self.RECORDS_NR_TRACK)

        logger.info(f"[STATS] Final: {stats}")

    def _update_task_statistics_location(self, db: Session, task: Task) -> None:
        """Update TaskStatisticsLocation count for task's location."""
        stats = db.query(TaskStatistics).filter(
            TaskStatistics.id == task.task_statistics_id
        ).first()

        if not stats:
            return

        # Find or create location record
        loc_record = db.query(TaskStatisticsLocation).filter(
            TaskStatisticsLocation.statistics_id == stats.id,
            TaskStatisticsLocation.location_id == task.location_id,
        ).first()

        if loc_record:
            loc_record.count = loc_record.count * 9 // 10 + 10
        else:
            loc_record = TaskStatisticsLocation(
                statistics_id=stats.id,
                location_id=task.location_id,
                count=10,
            )
            db.add(loc_record)

        logger.info(f"Updated TaskStatisticsLocation for: {task.name}")

    def _update_category_statistics(self, db: Session, task: Task) -> None:
        """Update CategoryStatistics for each category of the task."""
        # Get task categories
        from app.models import TaskCategory

        task_categories = db.query(TaskCategory).filter(
            TaskCategory.task_id == task.id
        ).all()

        for tc in task_categories:
            # Get category statistics
            cat_stats = db.query(CategoryStatistics).filter(
                CategoryStatistics.category_id == tc.category_id
            ).first()

            if not cat_stats:
                logger.warning(f"CategoryStatistics not found for category_id: {tc.category_id}")
                continue

            # Get current values
            current_records = cat_stats.records or 0
            current_difficulty = cat_stats.avg_difficulty or 0.0

            # EMA for avg_difficulty
            alpha = 1.0 / min(current_records + 1, self.RECORDS_NR_TRACK)
            new_difficulty = round((1 - alpha) * current_difficulty + alpha * task.difficulty, 2)
            cat_stats.avg_difficulty = new_difficulty

            # Update duration bucket
            bucket = self._get_duration_bucket(task.difficulty)
            avg_duration = copy.deepcopy(cat_stats.avg_duration) if cat_stats.avg_duration else {}

            if bucket not in avg_duration:
                avg_duration[bucket] = {"count": 0, "avg": 0}

            bucket_count = avg_duration[bucket].get("count", 0)
            bucket_avg = avg_duration[bucket].get("avg", 0)

            # EMA for duration bucket
            bucket_alpha = 1.0 / min(bucket_count + 1, self.RECORDS_NR_TRACK)
            new_bucket_avg = round((1 - bucket_alpha) * bucket_avg + bucket_alpha * task.duration, 2)
            avg_duration[bucket] = {
                "count": min(bucket_count + 1, self.RECORDS_NR_TRACK),
                "avg": int(new_bucket_avg)
            }
            cat_stats.avg_duration = avg_duration

            # Increment records (capped)
            cat_stats.records = min(current_records + 1, self.RECORDS_NR_TRACK)

            logger.info(f"[CAT_STATS] Final: {cat_stats}")

    def _update_category_statistics_location(self, db: Session, task: Task) -> None:
        """Update CategoryStatisticsLocation for each category of the task."""
        from app.models import TaskCategory, CategoryStatistics

        task_categories = db.query(TaskCategory).filter(
            TaskCategory.task_id == task.id
        ).all()

        for tc in task_categories:
            # Get category statistics
            cat_stats = db.query(CategoryStatistics).filter(
                CategoryStatistics.category_id == tc.category_id
            ).first()

            if not cat_stats:
                continue

            # Find or create location record
            loc_record = db.query(CategoryStatisticsLocation).filter(
                CategoryStatisticsLocation.statistics_id == cat_stats.id,
                CategoryStatisticsLocation.location_id == task.location_id,
            ).first()

            if loc_record:
                loc_record.count = loc_record.count * 9 // 10 + 10
            else:
                loc_record = CategoryStatisticsLocation(
                    statistics_id=cat_stats.id,
                    location_id=task.location_id,
                    count=10,
                )
                db.add(loc_record)

        logger.info(f"Updated CategoryStatisticsLocation for task: {task.name}")

    def update_time_score(
        self,
        db: Session,
        task_uuid: UUID,
        slot_start: datetime,
        boost: float,
    ) -> bool:
        """
        Update time scores with radial boost around a slot.
        
        Args:
            db: Database session
            task_uuid: UUID of the task
            slot_start: Datetime of the slot to boost
            boost: Boost value to apply (positive or negative)
        
        Returns:
            bool: True if successful, False on error
        """
        try:
            task = db.query(Task).filter(Task.id == task_uuid).first()
            if not task:
                logger.error(f"Task not found: {task_uuid}")
                return False

            if slot_start.tzinfo is not None:
                slot_start = slot_start.replace(tzinfo=None)

            target_hour = slot_start.hour
            target_minute = slot_start.minute
            target_minutes = target_hour * 60 + target_minute

            if task.task_statistics_id:
                self._update_statistics_time_scores(
                    db,
                    task.task_statistics_id,
                    target_minutes,
                    boost,
                    is_category=False,
                )

            task_categories = db.query(TaskCategory).filter(
                TaskCategory.task_id == task_uuid
            ).all()

            for tc in task_categories:
                cat_stats = db.query(CategoryStatistics).filter(
                    CategoryStatistics.category_id == tc.category_id
                ).first()
                if cat_stats:
                    self._update_statistics_time_scores(
                        db,
                        cat_stats.id,
                        target_minutes,
                        boost,
                        is_category=True,
                    )

            db.commit()
            logger.info(f"Time scores updated for task: {task.name}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update time scores for task {task_uuid}: {e}")
            return False

    def _update_statistics_time_scores(
        self,
        db: Session,
        statistics_id: UUID,
        target_minutes: int,
        boost: float,
        is_category: bool,
    ) -> None:
        """Update time_scores for a statistics record with radial boost."""
        if is_category:
            stats = db.query(CategoryStatistics).filter(
                CategoryStatistics.id == statistics_id
            ).first()
            time_scores_key = "category_time_scores"
        else:
            stats = db.query(TaskStatistics).filter(
                TaskStatistics.id == statistics_id
            ).first()
            time_scores_key = "task_time_scores"

        if not stats:
            return

        time_scores = copy.deepcopy(getattr(stats, time_scores_key)) if getattr(stats, time_scores_key) else {}

        offset = 0
        current_boost = boost

        while abs(current_boost) >= TIME_SCORE_STEP:
            if offset == 0:
                # Target slot: apply once
                slot_minutes = target_minutes
                hours = slot_minutes // 60
                minutes = slot_minutes % 60
                slot_key = f"{hours:02d}:{minutes:02d}"

                current_value = time_scores.get(slot_key, 0.0)
                new_value = current_value + current_boost
                new_value = max(TIME_SCORE_CLAMP[0], min(TIME_SCORE_CLAMP[1], new_value))
                time_scores[slot_key] = round(new_value, 2)
            else:
                # Adjacent slots: apply to both directions
                for sign in [-1, 1]:
                    slot_minutes = (target_minutes + sign * offset * 15) % (24 * 60)

                    hours = slot_minutes // 60
                    minutes = slot_minutes % 60
                    slot_key = f"{hours:02d}:{minutes:02d}"

                    current_value = time_scores.get(slot_key, 0.0)
                    new_value = current_value + current_boost
                    new_value = max(TIME_SCORE_CLAMP[0], min(TIME_SCORE_CLAMP[1], new_value))
                    time_scores[slot_key] = round(new_value, 2)

            offset += 1
            if boost >= 0:
                current_boost = boost - (offset * TIME_SCORE_STEP)
            else:
                current_boost = boost + (offset * TIME_SCORE_STEP)

        setattr(stats, time_scores_key, time_scores)

    def rate_task(
        self,
        db: Session,
        task_id: UUID,
        slot_start: datetime,
        completed: bool,
        actual_duration: Optional[int] = None,
        actual_difficulty: Optional[float] = None,
    ) -> bool:
        """
        Rate a task as completed or uncompleted.
        """
        try:
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                logger.error(f"Task not found: {task_id}")
                return False

            if slot_start.tzinfo is not None:
                slot_start = slot_start.replace(tzinfo=None)

            if completed:
                self._rate_completed(
                    db, task, slot_start, actual_duration, actual_difficulty
                )
                self.update_time_score(db, task_id, slot_start, boost=0.5)
            else:
                self._rate_uncompleted(db, task)
                self.update_time_score(db, task_id, slot_start, boost=-0.5)

            task.rated = True
            db.commit()
            logger.info(f"Task rated successfully: {task.name}, completed={completed}")
            return True

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to rate task {task_id}: {e}")
            return False

    def _rate_completed(
        self,
        db: Session,
        task: Task,
        slot_start: datetime,
        actual_duration: Optional[int],
        actual_difficulty: Optional[float],
    ) -> None:
        """Handle completed task rating."""
        difficulty_delta = actual_difficulty - task.difficulty
        duration_delta = actual_duration - task.duration
        bucket = self._get_duration_bucket(actual_difficulty)

        self._update_difficulty_delta(db, task.task_statistics_id, difficulty_delta, is_category=False)
        self._update_duration_delta(db, task.task_statistics_id, bucket, duration_delta, is_category=False)
        self._increment_completed_count(db, task.task_statistics_id, is_category=False)

        task_categories = db.query(TaskCategory).filter(
            TaskCategory.task_id == task.id
        ).all()
        for tc in task_categories:
            cat_stats = db.query(CategoryStatistics).filter(
                CategoryStatistics.category_id == tc.category_id
            ).first()
            if cat_stats:
                self._update_difficulty_delta(db, cat_stats.id, difficulty_delta, is_category=True)
                self._update_duration_delta(db, cat_stats.id, bucket, duration_delta, is_category=True)
                self._increment_completed_count(db, cat_stats.id, is_category=True)

    def _rate_uncompleted(self, db: Session, task: Task) -> None:
        """Handle uncompleted task rating."""
        if task.task_statistics_id:
            stats = db.query(TaskStatistics).filter(
                TaskStatistics.id == task.task_statistics_id
            ).first()
            if stats:
                stats.uncompleted_count = (stats.uncompleted_count or 0) + 1

        task_categories = db.query(TaskCategory).filter(
            TaskCategory.task_id == task.id
        ).all()
        for tc in task_categories:
            cat_stats = db.query(CategoryStatistics).filter(
                CategoryStatistics.category_id == tc.category_id
            ).first()
            if cat_stats:
                cat_stats.uncompleted_count = (cat_stats.uncompleted_count or 0) + 1

    def _update_difficulty_delta(
        self, db: Session, stats_id: UUID, difficulty_delta: float, is_category: bool
    ) -> None:
        """Update avg_difficulty_delta for stats record."""
        if is_category:
            stats = db.query(CategoryStatistics).filter(
                CategoryStatistics.id == stats_id
            ).first()
        else:
            stats = db.query(TaskStatistics).filter(
                TaskStatistics.id == stats_id
            ).first()

        if not stats:
            return

        current_avg_delta = stats.avg_difficulty_delta or 0.0
        current_completed = stats.completed_count or 0

        new_avg_delta = round(
            (current_avg_delta * current_completed + difficulty_delta) / (current_completed + 1),
            2
        )
        stats.avg_difficulty_delta = new_avg_delta

    def _update_duration_delta(
        self, db: Session, stats_id: UUID, bucket: str, duration_delta: int, is_category: bool
    ) -> None:
        """Update avg_duration_delta for stats record."""
        if is_category:
            stats = db.query(CategoryStatistics).filter(
                CategoryStatistics.id == stats_id
            ).first()
        else:
            stats = db.query(TaskStatistics).filter(
                TaskStatistics.id == stats_id
            ).first()

        if not stats:
            return

        avg_delta = copy.deepcopy(stats.avg_duration_delta) if stats.avg_duration_delta else {}

        if bucket not in avg_delta:
            avg_delta[bucket] = {"count": 0, "avg": 0}

        current_count = avg_delta[bucket].get("count", 0)
        current_avg = avg_delta[bucket].get("avg", 0)

        new_avg_delta = round(
            (current_avg * current_count + duration_delta) / (current_count + 1),
            2
        )
        avg_delta[bucket] = {
            "count": current_count + 1,
            "avg": int(new_avg_delta)
        }
        stats.avg_duration_delta = avg_delta

    def _increment_completed_count(self, db: Session, stats_id: UUID, is_category: bool) -> None:
        """Increment completed_count for stats record."""
        if is_category:
            stats = db.query(CategoryStatistics).filter(
                CategoryStatistics.id == stats_id
            ).first()
        else:
            stats = db.query(TaskStatistics).filter(
                TaskStatistics.id == stats_id
            ).first()

        if stats:
            stats.completed_count = (stats.completed_count or 0) + 1

    def _get_duration_bucket(self, difficulty: float) -> str:
        """Calculate duration bucket: round(difficulty * 2) / 2"""
        return str(round(difficulty * 2) / 2)


stats_recorder = StatsRecorder()