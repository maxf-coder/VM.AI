"""
Schedule Engine Service
Part of VM.AI pipeline - Stage 4

Handles:
- Constraint solving
- Slot generation
- Scoring
- Displacement handling

Version: 2.0 (Class-based)
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from uuid import UUID
from dataclasses import dataclass, field
from sqlalchemy.orm import Session

from app.models.task import Task
from app.models.schedule import ProvisionalSlot, MainScheduleSlot
from app.models.workflow import UnscheduledTask, ScheduleChange
from app.models.statistics import TaskStatistics, CategoryStatistics
from app.models.task_category import TaskCategory
from app.models.category import Category
from app.schemas.schedule import SchedulingResult, BatchSchedulingResult
from app.core.logging_config import setup_logging

logger = setup_logging()


# ============================================================================
# CONSTANTS (class-level)
# ============================================================================

SLOT_INTERVAL_MINUTES = 15
HORIZON_DAYS = 7
TOP_N_CANDIDATES = 400
VALUE_THRESHOLD = 1.25
MAX_DISPLACEMENT_LAYERS = 1
TIMEOUT_SECONDS = 12

LOCATION_BASE_BOOST = 0.25
FREE_SLOT_BOOST = 0.5
TIME_SCORE_AMPLIFIER = 0.3
URGENCY_AMPLIFIER = 0.3
CONTINUITY_BASE_BOOST = 0.1
OVERLAP_BASE_PENALTY = 0.15
BASE_SLOT_SCORE = 1.0

DEAD_ZONES = [
    ("23:00", "06:00"),
]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TimeWindow:
    """Represents a continuous time window for scheduling."""
    date: str
    start_dt: datetime
    end_dt: datetime

    @property
    def start_time(self) -> str:
        """Backward-compatible property returning HH:MM format."""
        return self.start_dt.strftime("%H:%M")

    @property
    def end_time(self) -> str:
        """Backward-compatible property returning HH:MM format."""
        return self.end_dt.strftime("%H:%M")


@dataclass
class CandidateSlot:
    """A candidate slot with score."""
    start: datetime
    end: datetime
    score: float


# ============================================================================
# SCHEDULE ENGINE CLASS
# ============================================================================

class ScheduleEngine:
    """
    Schedule engine service.
    
    Public Methods:
        schedule_single()   - Schedule one task
        schedule_batch() - Schedule multiple tasks
    """
    
    def __init__(self):
        self._old_start: Optional[datetime] = None
        self._old_end: Optional[datetime] = None
    
    # ============================================================================
    # PUBLIC METHODS
    # ============================================================================
    
    def schedule_single(
        self,
        task: Task,
        db: Session,
        create_change: bool = True,
        exclude_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
    ) -> SchedulingResult:
        """
        Schedule a single task into the provisional schedule.

        Uses savepoint to ensure provisional slot is restored on failure.
        If a task already exists in provisional_schedule, it is deleted
        before scheduling. On failure, the savepoint rollback restores
        the deleted slot.

        Args:
            task: Task to schedule
            db: Database session
            create_change: If False, skip creating ScheduleChange (for rescheduling)
            exclude_ranges: Optional list of (start, end) tuples to exclude from windows

        Returns:
            SchedulingResult with success status and slot info.
        """
        logger.info(f"Scheduling task: {task.name} (ID: {task.id})")

        # Savepoint: allows restoring deleted slot if scheduling fails
        sp = db.begin_nested()

        # Delete task's existing slot BEFORE overlap checks
        existing_slot = db.query(ProvisionalSlot).filter(
            ProvisionalSlot.task_id == task.id
        ).first()
        if existing_slot:
            logger.debug(f"Deleting existing slot for task {task.id} before scheduling")
            self._old_start = existing_slot.start
            self._old_end = existing_slot.end
            db.delete(existing_slot)
            db.flush()  # Make delete visible to subsequent queries
        else:
            self._old_start = self._old_end = None

        try:
            if task.fixed_time:
                result = self._schedule_fixed_task(task, db, create_change=create_change)
            else:
                result = self._schedule_flexible_task(task, db, create_change=create_change, exclude_ranges=exclude_ranges)

            if not result.success:
                sp.rollback()  # Restore deleted slot on failure
                return result

            return result
        except Exception as e:
            sp.rollback()
            db.rollback()
            logger.error(f"Scheduling failed for task {task.id}: {e}")
            return SchedulingResult(
                success=False,
                task_id=task.id,
                slot_id=None,
                slot_start=None,
                slot_end=None,
                displaced_tasks=[],
                message="Scheduling failed",
            )
    
    
    def schedule_batch(
        self,
        db: Session,
        timeout: int = TIMEOUT_SECONDS,
        task_ids: Optional[List[UUID]] = None,
    ) -> BatchSchedulingResult:
        """
        Schedule tasks from unscheduled_tasks queue or from provided list.

        If task_ids is None: query unscheduled_tasks queue
        If task_ids provided: use provided list

        Args:
            db: Database session
            timeout: Maximum execution time in seconds
            task_ids: Optional list of task IDs to schedule

        Returns:
            BatchSchedulingResult with scheduling results.
        """
        start_time = datetime.now()
        logger.info("Starting batch scheduling")
        
        if task_ids is not None:
            tasks = db.query(Task).filter(Task.id.in_(task_ids)).all()
            entries = tasks
            logger.info(f"Scheduling {len(task_ids)} provided task IDs")
        else:
            entries = (
                db.query(UnscheduledTask)
                .join(Task)
                .order_by(UnscheduledTask.created_at)
                .all()
            )
            logger.info(f"Scheduling {len(entries)} tasks from queue")
        
        results: List[Any] = []
        scheduled_count = 0
        failed_count = 0
        unscheduled_remaining: List[UUID] = []

        for entry in entries:
            if isinstance(entry, Task):
                task = entry
            else:
                task = entry.task

            if not task:
                logger.warning(f"Task not found")
                failed_count += 1
                unscheduled_remaining.append(entry.task_id if hasattr(entry, 'task_id') else None)
                continue

            result = self.schedule_single(task, db)
            results.append(result)

            if result.success:
                scheduled_count += 1
                if task_ids is None and hasattr(entry, 'task_id'):
                    db.delete(entry)
                db.commit()
            else:
                failed_count += 1
                unscheduled_remaining.append(task.id)

            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                logger.warning(f"Timeout reached after {elapsed:.1f}s")
                break

        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        logger.info(
            f"Batch complete: {scheduled_count} scheduled, {failed_count} failed, "
            f"{execution_time_ms}ms"
        )

        return BatchSchedulingResult(
            scheduled_count=scheduled_count,
            failed_count=failed_count,
            unscheduled_remaining=[uid for uid in unscheduled_remaining if uid is not None],
            results=results,
            execution_time_ms=execution_time_ms,
        )
    
    
    # ============================================================================
    # PRIVATE: FIXED TASK SCHEDULING
    # ============================================================================
    
    def _schedule_fixed_task(
        self,
        task: Task,
        db: Session,
        create_change: bool = True,
    ) -> SchedulingResult:
        """Schedule a fixed-time task at its fixed_start."""
        if not task.fixed_start or not task.duration:
            return SchedulingResult(
                success=False,
                task_id=task.id,
                slot_start=None,
                slot_end=None,
                displaced_tasks=[],
                message="Fixed task missing fixed_start or duration",
            )
        
        fixed_start = self._round_down_to_interval(task.fixed_start, SLOT_INTERVAL_MINUTES)
        fixed_end = self._calculate_fixed_task_end(task.fixed_start, task.duration)
        
        logger.debug(f"Fixed task at {fixed_start} - {fixed_end}")
        
        overlapping = self._get_overlapping_tasks(fixed_start, fixed_end, db)
        
        displaced_ids: List[UUID] = []
        
        for existing in overlapping:
            if existing.fixed:
                return SchedulingResult(
                    success=False,
                    task_id=task.id,
                    slot_start=fixed_start,
                    slot_end=fixed_end,
                    displaced_tasks=[],
                    message=f"Slot occupied by fixed task {existing.task_id}",
                )
            
            existing_task = db.query(Task).filter(Task.id == existing.task_id).first()
            if existing_task and existing_task.fixed_time:
                return SchedulingResult(
                    success=False,
                    task_id=task.id,
                    slot_start=fixed_start,
                    slot_end=fixed_end,
                    displaced_tasks=[],
                    message=f"Slot occupied by fixed task {existing.task_id}",
                )
            
            displaced_id = existing.task_id
            
            old_start = existing.start
            old_end = existing.end
            
            db.delete(existing)
            
            existing_task_for_displacement = db.query(Task).filter(Task.id == displaced_id).first()
            if existing_task_for_displacement:
                # Pass fixed task's time range to exclude when rescheduling
                exclude_ranges = [(fixed_start, fixed_end)]
                reschedule_result = self._try_reschedule_task(existing_task_for_displacement, db, layer=1, exclude_ranges=exclude_ranges)

                if not reschedule_result.success:
                    db.rollback()
                    return SchedulingResult(
                        success=False,
                        task_id=task.id,
                        slot_start=fixed_start,
                        slot_end=fixed_end,
                        displaced_tasks=[],
                        message=f"Cannot displace {displaced_id}: cannot be rescheduled",
                    )

                new_change = ScheduleChange(
                    provisional_schedule_slot_id=reschedule_result.slot_id,
                    change_type="move",
                    old_slot_start=old_start,
                    old_slot_end=old_end,
                    new_slot_start=reschedule_result.slot_start,
                    new_slot_end=reschedule_result.slot_end,
                )
                db.add(new_change)
                displaced_ids.append(displaced_id)
            else:
                logger.warning(f"Task {displaced_id} not found for displacement")

        return self._place_task(task, fixed_start, fixed_end, db, displaced_ids, create_change=create_change)
    
    
    # ============================================================================
    # PRIVATE: FLEXIBLE TASK SCHEDULING
    # ============================================================================
    
    def _schedule_flexible_task(
        self,
        task: Task,
        db: Session,
        create_change: bool = True,
        exclude_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
    ) -> SchedulingResult:
        """Schedule a flexible task with full scheduling pipeline."""
        logger.debug(f"Flexible task: {task.name}")

        # Pre-fetch all provisional slots once for in-memory scoring
        all_slots = db.query(ProvisionalSlot).all()
        fixed_slots = [s for s in all_slots if s.fixed]

        windows = self._get_time_windows(task, db, exclude_ranges=exclude_ranges, fixed_slots=fixed_slots)
        if not windows:
            return SchedulingResult(
                success=False,
                task_id=task.id,
                slot_start=None,
                slot_end=None,
                displaced_tasks=[],
                message="No viable time windows",
            )
        
        duration = task.duration or 30
        slots = self._generate_slots(windows, duration)
        if not slots:
            return SchedulingResult(
                success=False,
                task_id=task.id,
                slot_start=None,
                slot_end=None,
                displaced_tasks=[],
                message="No viable slots after generation",
            )

        stats_cache = self._prefetch_scoring_data(task, db)

        scored_slots = []
        for slot_start in slots:
            slot_end = self._calculate_end(slot_start, duration)
            score = self._score_slot(slot_start, slot_end, task, all_slots, stats_cache)
            scored_slots.append(CandidateSlot(slot_start, slot_end, score))
        
        scored_slots.sort(key=lambda s: s.score, reverse=True)
        top_slots = scored_slots[:TOP_N_CANDIDATES]
        
        if not top_slots:
            return SchedulingResult(
                success=False,
                task_id=task.id,
                slot_start=None,
                slot_end=None,
                displaced_tasks=[],
                message="No candidate slots after scoring",
            )
        
        for candidate in top_slots:
            result = self._handle_displacement(task, candidate.start, candidate.end, db, layer=1, create_change=create_change)
            if result.success:
                logger.info(f"Task scheduled at {candidate.start}")
                return result
        
        return SchedulingResult(
            success=False,
            task_id=task.id,
            slot_start=None,
            slot_end=None,
            displaced_tasks=[],
            message="Could not find viable slot",
        )
    
    
    # ============================================================================
    # PRIVATE: RESCHEDULE
    # ============================================================================
    
    def _try_reschedule_task(
        self,
        task: Task,
        db: Session,
        layer: int = 1,
        exclude_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
    ) -> SchedulingResult:
        """Try to reschedule a displaced task."""
        if layer > MAX_DISPLACEMENT_LAYERS:
            return SchedulingResult(
                success=False,
                task_id=task.id,
                slot_start=None,
                slot_end=None,
                displaced_tasks=[],
                message=f"Max displacement layers ({MAX_DISPLACEMENT_LAYERS}) exceeded",
            )

        # Rescheduling creates slot but NOT change - the displacer creates the "move" change
        return self.schedule_single(task, db, create_change=False, exclude_ranges=exclude_ranges)
    
    
    # ============================================================================
    # PRIVATE: CONSTRAINT SOLVER
    # ============================================================================
    
    def _get_time_windows(
        self,
        task: Task,
        db: Session,
        exclude_ranges: Optional[List[Tuple[datetime, datetime]]] = None,
        fixed_slots: Optional[List[ProvisionalSlot]] = None,
    ) -> Dict[str, List[TimeWindow]]:
        """Build viable time windows for task.
        
        Args:
            task: Task to schedule
            db: Database session
            exclude_ranges: Optional list of (start, end) datetime tuples to exclude
            fixed_slots: Optional pre-fetched list of fixed ProvisionalSlots
        """
        windows: Dict[str, List[TimeWindow]] = {}
        
        now = datetime.now()
        
        if task.start:
            task_start = task.start.replace(tzinfo=None) if task.start.tzinfo else task.start
        else:
            task_start = now
        
        if task.deadline:
            task_deadline = task.deadline.replace(tzinfo=None) if task.deadline.tzinfo else task.deadline
        else:
            task_deadline = task_start + timedelta(days=7)
        
        horizon_end = now + timedelta(days=HORIZON_DAYS)
        if task_deadline > horizon_end:
            task_deadline = horizon_end
        
        current = task_start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = task_deadline.replace(hour=0, minute=0, second=0, microsecond=0)
        is_first_day = True
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            
            if is_first_day:
                start_dt = task_start
                is_first_day = False
            else:
                start_dt = datetime.strptime(f"{date_str} 00:00", "%Y-%m-%d %H:%M")
            
            if current.date() == task_deadline.date():
                end_dt = task_deadline
            else:
                end_dt = datetime.strptime(f"{date_str} 23:59", "%Y-%m-%d %H:%M")
            
            windows[date_str] = [TimeWindow(date=date_str, start_dt=start_dt, end_dt=end_dt)]
            current += timedelta(days=1)
        
        if exclude_ranges:
            logger.debug(f"Excluding {len(exclude_ranges)} ranges from windows")
            for excl_start, excl_end in exclude_ranges:
                logger.debug(f"  Excluding: {excl_start.strftime('%H:%M')}-{excl_end.strftime('%H:%M')}")
                excl_date_str = excl_start.strftime("%Y-%m-%d")
                if excl_date_str in windows:
                    new_windows = []
                    for window in windows[excl_date_str]:
                        if excl_start >= window.end_dt or excl_end <= window.start_dt:
                            new_windows.append(window)
                        else:
                            if window.start_dt < excl_start:
                                new_windows.append(TimeWindow(
                                    date=excl_date_str,
                                    start_dt=window.start_dt,
                                    end_dt=excl_start,
                                ))
                            if excl_end < window.end_dt:
                                new_windows.append(TimeWindow(
                                    date=excl_date_str,
                                    start_dt=excl_end,
                                    end_dt=window.end_dt,
                                ))
                    windows[excl_date_str] = new_windows
        
        windows = self._subtract_fixed_tasks(windows, db, fixed_slots=fixed_slots)
        
        windows = self._subtract_dead_zones(windows)
        
        valid_windows = {}
        for date_str, ws in windows.items():
            for w in ws:
                if w.end_dt > w.start_dt:
                    if date_str not in valid_windows:
                        valid_windows[date_str] = []
                    valid_windows[date_str].append(w)
        
        for ds, ws in valid_windows.items():
            logger.debug(f"  Date {ds}: {len(ws)} windows")
        
        return valid_windows
    
    
    def _subtract_fixed_tasks(
        self,
        windows: Dict[str, List[TimeWindow]],
        db: Session,
        fixed_slots: Optional[List[ProvisionalSlot]] = None,
    ) -> Dict[str, List[TimeWindow]]:
        """Subtract slots occupied by fixed tasks."""
        if fixed_slots is not None:
            fixed_tasks = fixed_slots
        else:
            fixed_tasks = db.query(ProvisionalSlot).filter(ProvisionalSlot.fixed == True).all()

        for slot in fixed_tasks:
            date_str = slot.start.strftime("%Y-%m-%d")
            if date_str in windows:
                new_windows = []
                for window in windows[date_str]:
                    # Compare datetimes directly (both are datetime objects now)
                    if slot.start >= window.end_dt or slot.end <= window.start_dt:
                        new_windows.append(window)
                    else:
                        if window.start_dt < slot.start:
                            new_windows.append(TimeWindow(
                                date=date_str,
                                start_dt=window.start_dt,
                                end_dt=slot.start,
                            ))
                        if slot.end < window.end_dt:
                            new_windows.append(TimeWindow(
                                date=date_str,
                                start_dt=slot.end,
                                end_dt=window.end_dt,
                            ))
                windows[date_str] = new_windows

        return windows
    
    
    def _subtract_dead_zones(
        self,
        windows: Dict[str, List[TimeWindow]],
    ) -> Dict[str, List[TimeWindow]]:
        """Subtract dead zones from time windows."""
        for zone_start_str, zone_end_str in DEAD_ZONES:
            for date_str in windows:
                zs = datetime.strptime(f"{date_str} {zone_start_str}", "%Y-%m-%d %H:%M")
                ze = datetime.strptime(f"{date_str} {zone_end_str}", "%Y-%m-%d %H:%M")

                new_windows = []
                for window in windows[date_str]:
                    ws = window.start_dt
                    we = window.end_dt

                    # Check if dead zone spans midnight (e.g., 23:00-06:00)
                    if ze < ws:  # Zone spans midnight (zone_end < zone_start)
                        # Valid time is ONLY: zone_end (06:00) to zone_start (23:00)
                        # Skip morning dead zone: if window starts before ze, start at ze
                        if ws < ze:
                            ws = ze
                        # Skip evening dead zone: if window ends after zs, end at zs
                        if we > zs:
                            we = zs

                        # Only add window if valid range exists
                        if ws < we:
                            new_windows.append(TimeWindow(
                                date=date_str,
                                start_dt=ws,
                                end_dt=we,
                            ))
                    else:
                        # Normal case: dead zone doesn't span midnight (e.g., 09:00-11:00)
                        if we <= zs or ws >= ze:
                            new_windows.append(window)
                        else:
                            if ws < zs:
                                new_windows.append(TimeWindow(
                                    date=date_str,
                                    start_dt=ws,
                                    end_dt=zs,
                                ))
                            if we > ze:
                                new_windows.append(TimeWindow(
                                    date=date_str,
                                    start_dt=ze,
                                    end_dt=we,
                                ))
                windows[date_str] = new_windows

        return windows
    
    
    # ============================================================================
    # PRIVATE: SLOT GENERATOR
    # ============================================================================
    
    def _generate_slots(
        self,
        windows: Dict[str, List[TimeWindow]],
        duration_minutes: int,
    ) -> List[datetime]:
        """Split windows into 15-min slot start times."""
        slots: List[datetime] = []

        for date_str, ws in windows.items():
            for window in ws:
                start_dt = window.start_dt
                end_dt = window.end_dt

                window_end_total_minutes = end_dt.hour * 60 + end_dt.minute
                duration_total_minutes = self._round_up_to_interval(duration_minutes)

                deadzone_start_minutes = 23 * 60
                valid_end_minutes = min(window_end_total_minutes - duration_total_minutes, deadzone_start_minutes - duration_total_minutes)

                if valid_end_minutes <= 0:
                    continue

                current_dt = start_dt

                while (current_dt.hour * 60 + current_dt.minute) <= valid_end_minutes:
                    slots.append(current_dt)

                    # Add SLOT_INTERVAL_MINUTES to current_dt
                    new_minute = current_dt.minute + SLOT_INTERVAL_MINUTES
                    new_hour = current_dt.hour + new_minute // 60
                    new_minute = new_minute % 60
                    current_dt = current_dt.replace(hour=new_hour, minute=new_minute)
        
        slots.sort()
        return slots
    
    
    # ============================================================================
    # PRIVATE: SCORER
    # ============================================================================
    
    def _score_slot(
        self,
        slot_start: datetime,
        slot_end: datetime,
        task: Task,
        all_slots: List[ProvisionalSlot],
        stats_cache: dict,
    ) -> float:
        """Calculate total score for a slot (no DB queries)."""
        score = BASE_SLOT_SCORE

        score += self._get_location_boost(slot_start, task, all_slots)
        score += self._get_free_slot_boost(slot_start, slot_end, all_slots)
        score += self._get_time_score_boost(slot_start, stats_cache)
        score += self._get_urgency_boost(slot_start, task)
        score += self._get_continuity_boost(slot_start, all_slots)
        score -= self._get_overlap_penalty(slot_start, slot_end, all_slots)

        return score
    
    
    def _get_location_boost(
        self,
        slot_start: datetime,
        task: Task,
        all_slots: List[ProvisionalSlot],
    ) -> float:
        """Calculate location continuity boost (in-memory)."""
        if not task.location:
            return 0.0

        task_location = task.location.name

        before = None
        after = None

        for s in all_slots:
            if s.end <= slot_start and s.end > slot_start - timedelta(hours=2):
                if before is None or s.end > before.end:
                    before = s
            if s.start >= slot_start and s.start < slot_start + timedelta(hours=2):
                if after is None or s.start < after.start:
                    after = s

        continuity_count = 0

        if before and before.location == task_location:
            continuity_count += 0.5

        if after and after.location == task_location:
            continuity_count += 0.5

        return LOCATION_BASE_BOOST * continuity_count
    
    
    def _get_free_slot_boost(
        self,
        slot_start: datetime,
        slot_end: datetime,
        all_slots: List[ProvisionalSlot],
    ) -> float:
        """Calculate free slot boost (in-memory)."""
        overlapping = self._filter_overlapping(slot_start, slot_end, all_slots)

        if not overlapping:
            return FREE_SLOT_BOOST

        return 0.0
    
    
    def _get_time_score_boost(
        self,
        slot_start: datetime,
        stats_cache: dict,
    ) -> float:
        """Calculate time preference score boost (using cached stats)."""
        time_key = slot_start.strftime("%H:%M")

        task_stats = stats_cache.get("task_stats")
        if task_stats and task_stats.task_time_scores and task_stats.records > 3:
            if time_key in task_stats.task_time_scores:
                score = task_stats.task_time_scores[time_key]
                return TIME_SCORE_AMPLIFIER * (score / 10)

        assoc_stats = stats_cache.get("assoc_stats")
        if assoc_stats and assoc_stats.task_time_scores and assoc_stats.records > 3:
            if time_key in assoc_stats.task_time_scores:
                score = assoc_stats.task_time_scores[time_key]
                return TIME_SCORE_AMPLIFIER * (score / 10)

        for tc in stats_cache.get("task_cats", []):
            cat_stats = stats_cache.get("cat_stats", {}).get(tc.category_id)
            if cat_stats and cat_stats.category_time_scores:
                if time_key in cat_stats.category_time_scores:
                    score = cat_stats.category_time_scores[time_key]
                    return TIME_SCORE_AMPLIFIER * (score / 10)

        return 0.0
    
    
    def _get_urgency_boost(
        self,
        slot_start: datetime,
        task: Task,
    ) -> float:
        """Calculate urgency boost (earlier slots get higher boost)."""
        total_minutes = HORIZON_DAYS * 24 * 60
        
        now = datetime.now()
        
        if slot_start.tzinfo:
            slot_start = slot_start.replace(tzinfo=None)
        
        if slot_start <= now:
            logger.warning(f"Slot {slot_start} is in the past (now: {now})")
            return -1.0
        
        slot_minutes = (slot_start - now).total_seconds() / 60
        
        position_ratio = slot_minutes / total_minutes
        
        urgency_value = (task.urgency or 0.5) * 0.5 + (task.importance or 0.5) * 0.5
        
        return URGENCY_AMPLIFIER * urgency_value * (1 - position_ratio)
    
    
    def _get_continuity_boost(
        self,
        slot_start: datetime,
        all_slots: List[ProvisionalSlot],
    ) -> float:
        """Calculate proximity boost to previous task (in-memory)."""
        if slot_start.tzinfo:
            slot_start = slot_start.replace(tzinfo=None)

        prev_task = None
        for s in all_slots:
            if s.end <= slot_start:
                if prev_task is None or s.end > prev_task.end:
                    prev_task = s

        if not prev_task:
            return 0.0

        prev_end = prev_task.end.replace(tzinfo=None) if prev_task.end.tzinfo else prev_task.end
        minutes_diff = (slot_start - prev_end).total_seconds() / 60

        slots_diff = round(minutes_diff / SLOT_INTERVAL_MINUTES)

        if slots_diff <= 0:
            return 0.0
        elif slots_diff == 1:
            return CONTINUITY_BASE_BOOST * 0.5
        elif slots_diff == 2:
            return CONTINUITY_BASE_BOOST * 1.0
        elif slots_diff == 3:
            return CONTINUITY_BASE_BOOST * 0.5
        else:
            return 0.0
    
    
    def _get_overlap_penalty(
        self,
        slot_start: datetime,
        slot_end: datetime,
        all_slots: List[ProvisionalSlot],
    ) -> float:
        """Calculate overlap penalty (in-memory)."""
        overlapping = self._filter_overlapping(slot_start, slot_end, all_slots)

        if not overlapping:
            return 0.0

        return OVERLAP_BASE_PENALTY * len(overlapping)
    
    
    # ============================================================================
    # PRIVATE: DISPLACEMENT HANDLER
    # ============================================================================
    
    def _handle_displacement(
        self,
        task: Task,
        slot_start: datetime,
        slot_end: datetime,
        db: Session,
        layer: int = 1,
        create_change: bool = True,
    ) -> SchedulingResult:
        """Try to place task in slot, displacing if needed."""
        overlapping = self._get_overlapping_tasks(slot_start, slot_end, db)
        
        if not overlapping:
            return self._place_task(task, slot_start, slot_end, db, [], create_change=create_change)
        
        fixed_tasks = [t for t in overlapping if t.fixed]
        if fixed_tasks:
            return SchedulingResult(
                success=False,
                task_id=task.id,
                slot_start=slot_start,
                slot_end=slot_end,
                displaced_tasks=[],
                message="Slot occupied by fixed task",
            )
        
        sorted_tasks = sorted(overlapping, key=lambda t: t.value or 0.0)
        displaced_ids: List[UUID] = []
        
        for existing in sorted_tasks:
            if not self._can_displace(task, existing):
                return SchedulingResult(
                    success=False,
                    task_id=task.id,
                    slot_start=slot_start,
                    slot_end=slot_end,
                    displaced_tasks=displaced_ids,
                    message=f"Cannot displace {existing.task_id}: value threshold not met",
                )
            
            displaced_task = db.query(Task).filter(Task.id == existing.task_id).first()
            if not displaced_task:
                logger.warning(f"Task {existing.task_id} not found for displacement")
                continue
            
            old_start = existing.start
            old_end = existing.end
            
            db.delete(existing)
            
            # Pass the displacer's time range to exclude when rescheduling
            exclude_ranges = [(slot_start, slot_end)]
            reschedule_result = self._try_reschedule_task(displaced_task, db, layer=layer + 1, exclude_ranges=exclude_ranges)
            
            if not reschedule_result.success:
                db.rollback()
                return SchedulingResult(
                    success=False,
                    task_id=task.id,
                    slot_start=slot_start,
                    slot_end=slot_end,
                    displaced_tasks=displaced_ids,
                    message=f"Cannot displace {existing.task_id}: cannot be rescheduled",
                )
            
            new_change = ScheduleChange(
                provisional_schedule_slot_id=reschedule_result.slot_id,
                change_type="move",
                old_slot_start=old_start,
                old_slot_end=old_end,
                new_slot_start=reschedule_result.slot_start,
                new_slot_end=reschedule_result.slot_end,
            )
            db.add(new_change)
            displaced_ids.append(existing.task_id)
        
        return self._place_task(task, slot_start, slot_end, db, displaced_ids, create_change=create_change)
    
    
    def _can_displace(self, new_task: Task, existing: ProvisionalSlot) -> bool:
        """Check if new_task can displace existing task."""
        existing_value = existing.value or 0.0
        new_value = new_task.value or 0.0
        
        return new_value > existing_value * VALUE_THRESHOLD
    
    
    def _place_task(
        self,
        task: Task,
        slot_start: datetime,
        slot_end: datetime,
        db: Session,
        displaced_ids: List[UUID],
        create_change: bool = True,
    ) -> SchedulingResult:
        """Place task in provisional schedule."""
        # Slot already deleted in schedule_single() — just insert
        slot = ProvisionalSlot(
            task_id=task.id,
            start=slot_start,
            end=slot_end,
            value=task.value,
            fixed=task.fixed_time,
            location=task.location.name if task.location else None,
        )
        db.add(slot)
        db.flush()

        if create_change:
            change = ScheduleChange(
                provisional_schedule_slot_id=slot.id,
                change_type="move" if self._old_start else "insert",
                old_slot_start=self._old_start,
                old_slot_end=self._old_end,
                new_slot_start=slot_start,
                new_slot_end=slot_end,
            )
            db.add(change)
        db.commit()

        return SchedulingResult(
            success=True,
            task_id=task.id,
            slot_id=slot.id,
            slot_start=slot_start,
            slot_end=slot_end,
            displaced_tasks=displaced_ids,
            message="Task scheduled successfully",
        )
    
    
    # ============================================================================
    # PRIVATE: UTILITIES
    # ============================================================================
    
    def _get_overlapping_tasks(
        self,
        slot_start: datetime,
        slot_end: datetime,
        db: Session,
    ) -> List[ProvisionalSlot]:
        """Get all slots that overlap with the given time range."""
        result = db.query(ProvisionalSlot).filter(
            ProvisionalSlot.start < slot_end,
            ProvisionalSlot.end > slot_start,
        ).all()
        return result


    @staticmethod
    def _filter_overlapping(
        slot_start: datetime,
        slot_end: datetime,
        slots: List[ProvisionalSlot],
    ) -> List[ProvisionalSlot]:
        """Filter pre-fetched slots overlapping with the given time range (in-memory)."""
        return [
            s for s in slots
            if s.start < slot_end and s.end > slot_start
        ]


    def _prefetch_scoring_data(
        self,
        task: Task,
        db: Session,
    ) -> dict:
        """Pre-fetch task statistics and category stats for scoring (one-time per task)."""
        cache = {
            "task_stats": None,
            "assoc_stats": None,
            "task_cats": [],
            "cat_stats": {},
        }

        if task.task_statistics_id:
            cache["task_stats"] = db.query(TaskStatistics).filter(
                TaskStatistics.id == task.task_statistics_id
            ).first()

        if task.associated_task_statistics_id:
            cache["assoc_stats"] = db.query(TaskStatistics).filter(
                TaskStatistics.id == task.associated_task_statistics_id
            ).first()

        task_cats = (
            db.query(TaskCategory)
            .filter(TaskCategory.task_id == task.id)
            .order_by(TaskCategory.priority)
            .all()
        )
        cache["task_cats"] = task_cats

        for tc in task_cats:
            cat_stats = db.query(CategoryStatistics).filter(
                CategoryStatistics.category_id == tc.category_id
            ).first()
            if cat_stats:
                cache["cat_stats"][tc.category_id] = cat_stats

        return cache
    
    
    def _round_up_to_interval(self, minutes: int) -> int:
        """Round minutes up to nearest slot interval."""
        return ((minutes + SLOT_INTERVAL_MINUTES - 1) // SLOT_INTERVAL_MINUTES) * SLOT_INTERVAL_MINUTES
    
    
    def _round_down_to_interval(self, dt: datetime, interval: int) -> datetime:
        """Round datetime down to nearest interval."""
        total_minutes = dt.hour * 60 + dt.minute
        rounded = (total_minutes // interval) * interval
        return dt.replace(hour=rounded // 60, minute=rounded % 60, second=0, microsecond=0)
    
    
    def _calculate_end(self, start: datetime, duration_minutes: int) -> datetime:
        """Calculate end time from start and duration."""
        rounded_duration = self._round_up_to_interval(duration_minutes)
        return start + timedelta(minutes=rounded_duration)
    
    
    def _calculate_fixed_task_end(self, fixed_start: datetime, duration_minutes: int) -> datetime:
        """Calculate end time for fixed task."""
        raw_end = fixed_start + timedelta(minutes=duration_minutes)
        return self._round_up_to_interval_dt(raw_end)
    
    
    def _round_up_to_interval_dt(self, dt: datetime) -> datetime:
        """Round datetime UP to nearest interval."""
        total_minutes = dt.hour * 60 + dt.minute
        rounded = ((total_minutes + SLOT_INTERVAL_MINUTES - 1) // SLOT_INTERVAL_MINUTES) * SLOT_INTERVAL_MINUTES
        if rounded >= 24 * 60:
            return dt.replace(hour=23, minute=59, second=59, microsecond=0)
        return dt.replace(hour=rounded // 60, minute=rounded % 60, second=0, microsecond=0)

    
    
# ============================================================================
# EXPORT SINGLETON INSTANCE
# ============================================================================

schedule_engine = ScheduleEngine()