import logging
from typing import Optional, List, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from app.models import Task, Location, Category, TaskCategory, TaskStatistics, CategoryStatistics
from app.models.workflow import UnscheduledTask
from app.schemas.enrichment import TaskPayloadComputedWithRefs, TaskPayloadComputed
from app.core.logging_config import setup_logging

logger = setup_logging()

DEFAULT_TIME_ANCHORS: dict[str, float] = {
    "00:00": -1.5,
    "02:00": -2,
    "06:00": -1,
    "09:00": 0.5,
    "12:00": 1.5,
    "14:00": 2,
    "16:00": 1.25,
    "18:00": 1.0,
    "20:00": 0.5,
    "22:00": 0.0,
    "23:00": -1.0,
}


def generate_default_time_scores(
    anchors: dict[str, float],
    step: float = 0.25
) -> dict[str, float]:
    """
    Generate time scores using linear interpolation between anchor points.
    
    Args:
        anchors: Dict of time strings to values, e.g. {"09:00": 0.0, "15:00": 1.5}
        step: Interval step in hours (default 0.25 = 15 minutes)
    
    Returns:
        Dict of time strings to interpolated float values, e.g. {"09:00": 0.0, "09:15": 0.375, ...}
    """
    if not anchors:
        return {}
    
    def time_to_minutes(t: str) -> int:
        parts = t.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    
    def minutes_to_time(m: int) -> str:
        hours = (m // 60) % 24
        mins = m % 60
        return f"{hours:02d}:{mins:02d}"
    
    sorted_anchors = sorted(anchors.items(), key=lambda x: time_to_minutes(x[0]))
    anchor_minutes = [(time_to_minutes(t), v) for t, v in sorted_anchors]
    
    if len(anchor_minutes) < 2:
        return {}
    
    first_min = anchor_minutes[0][0]
    last_min = anchor_minutes[-1][0]
    
    time_scores: dict[str, float] = {}
    MAX_DAY_END = 23 * 60 + 45  # 1425 minutes = 23:45
    
    if last_min > first_min:
        start = first_min
        last_anchor_value = anchor_minutes[-1][1]
        end = MAX_DAY_END  # Always extend to 23:45
        while start <= end:
            time_str = minutes_to_time(start)
            if start > last_min:
                rounded_value = round(last_anchor_value / step) * step
            else:
                value = _interpolate_value(start, anchor_minutes)
                rounded_value = round(value / step) * step
            time_scores[time_str] = rounded_value
            start += int(step * 60)
    else:
        for m in range(0, 24 * 60, int(step * 60)):
            time_str = minutes_to_time(m)
            value = _interpolate_circular(m, anchor_minutes)
            rounded_value = round(value / step) * step
            time_scores[time_str] = rounded_value
    
    return time_scores


def _interpolate_value(current_min: int, anchors: list[tuple[int, float]]) -> float:
    """Linear interpolation between two anchor points."""
    for i in range(len(anchors) - 1):
        t1, v1 = anchors[i]
        t2, v2 = anchors[i + 1]
        if t1 <= current_min <= t2:
            if t2 == t1:
                return v1
            ratio = (current_min - t1) / (t2 - t1)
            return v1 + (v2 - v1) * ratio
    return anchors[-1][1]


def _interpolate_circular(current_min: int, anchors: list[tuple[int, float]]) -> float:
    """Circular interpolation for times that span midnight."""
    first_min = anchors[0][0]
    last_min = anchors[-1][0]
    
    if first_min <= current_min <= last_min:
        return _interpolate_value(current_min, anchors)
    
    if current_min < first_min:
        t1, v1 = anchors[-1]
        t2, v2 = anchors[0]
        span = (24 * 60 - t1) + t2
        if span == 0:
            return v1
        ratio = ((24 * 60 - current_min) - (24 * 60 - t1)) / span
        ratio = 1 - ratio
        return v1 + (v2 - v1) * ratio
    
    return anchors[0][1]


def save_commited_task(db: Session, enriched_task: TaskPayloadComputedWithRefs) -> Task | None:
    """
    Save enriched task to DB tables using ORM.
    
    Returns True on success, False on failure (with rollback).
    """
    try:
        location = _ensure_location(db, enriched_task.location)
        
        task_stats_ids = _handle_task_statistics(db, enriched_task)
        
        task = Task(
            task_statistics_id=task_stats_ids[0],
            associated_task_statistics_id=task_stats_ids[1],
            name=enriched_task.name,
            start=enriched_task.start,
            deadline=enriched_task.deadline,
            difficulty=enriched_task.difficulty,
            duration=enriched_task.duration,
            location_id=location.id,
            importance=enriched_task.importance,
            urgency=enriched_task.urgency,
            value=enriched_task.value,
            fixed_time=enriched_task.fixed_time,
            fixed_start=enriched_task.fixed_start,
            rated=False,
        )
        db.add(task)
        db.flush()
        
        # Also add to unscheduled_tasks queue
        unscheduled = UnscheduledTask(task_id=task.id)
        db.add(unscheduled)
        
        for priority, cat_name in enumerate(enriched_task.category):
            category = _ensure_category(db, cat_name)
            db.add(TaskCategory(
                task_id=task.id,
                category_id=category.id,
                priority=priority
            ))
            # Ensure category statistics exists
            _ensure_category_statistics(db, cat_name)
        
        db.commit()
        logger.info(f"Task saved successfully: {enriched_task.name}")
        return task
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save task '{enriched_task.name}': {e}")
        return None


def _ensure_location(db: Session, location_name: str) -> Location:
    """Ensure location exists in DB, create if needed. Returns Location object."""
    location = db.query(Location).filter(Location.name == location_name).first()
    
    if not location:
        location = Location(name=location_name)
        db.add(location)
        db.flush()
        logger.info(f"Created new location: {location_name}")
    
    return location


def _ensure_category(db: Session, category_name: str) -> Category:
    """Ensure category exists in DB, create if needed. Returns Category object."""
    category = db.query(Category).filter(Category.name == category_name).first()
    
    if not category:
        category = Category(name=category_name)
        db.add(category)
        db.flush()
        logger.info(f"Created new category: {category_name}")
    
    return category


def _handle_task_statistics(
    db: Session,
    enriched_task: TaskPayloadComputedWithRefs
) -> Tuple[UUID, Optional[UUID]]:
    """
    Handle task_statistics based on association_status.
    
    Returns (task_statistics_id, associated_task_statistics_id) for Task table.
    
    Logic from notes.log:
    - "same": task_statistics_id = associated_id, associated_task_statistics_id = None
    - "similar": task_statistics_id = new row's id, associated_task_statistics_id = associated_id
    - "none": task_statistics_id = new row's id, associated_task_statistics_id = None
    """
    association_status = enriched_task.association_status
    associated_id = enriched_task.task_statistics_id
    
    if association_status == "same" and associated_id:
        return (associated_id, None)
    
    new_stats = _create_task_statistics(
        db,
        enriched_task.name,
        enriched_task.name_vector
    )
    
    if association_status == "similar" and associated_id:
        return (new_stats.id, associated_id)
    
    return (new_stats.id, None)


def _create_task_statistics(
    db: Session,
    task_name: str,
    name_vector: Optional[List[float]] = None
) -> TaskStatistics:
    """Create new TaskStatistics record. Returns TaskStatistics object."""
    stats = TaskStatistics(
        task_name=task_name,
        task_name_vector=name_vector,
        avg_duration={},
        avg_duration_delta={},
        avg_difficulty=0.0,
        avg_difficulty_delta=0.0,
        completed_count=0,
        uncompleted_count=0,
        records=0,
        task_time_scores=generate_default_time_scores(DEFAULT_TIME_ANCHORS),
    )
    db.add(stats)
    db.flush()
    logger.info(f"Created new task_statistics: {task_name}")
    return stats


def _ensure_category_statistics(
    db: Session,
    category_name: str,
) -> Optional[CategoryStatistics]:
    """
    Ensure category statistics exists in DB, create if needed.
    Returns CategoryStatistics object or None if category not found.
    """
    # Get category by name
    category = db.query(Category).filter(Category.name == category_name).first()
    
    if not category:
        logger.warning(f"Category not found: {category_name}")
        return None
    
    # Check if statistics already exists
    stats = db.query(CategoryStatistics).filter(
        CategoryStatistics.category_id == category.id
    ).first()
    
    if stats:
        return stats
    
    # Create new CategoryStatistics
    stats = CategoryStatistics(
        category_id=category.id,
        avg_duration={},
        avg_duration_delta={},
        avg_difficulty=0.0,
        avg_difficulty_delta=0.0,
        completed_count=0,
        uncompleted_count=0,
        records=0,
        category_time_scores=generate_default_time_scores(DEFAULT_TIME_ANCHORS),
    )
    db.add(stats)
    db.flush()
    logger.info(f"Created new category_statistics: {category_name}")
    return stats


def update_commited_task(db: Session, task_id: UUID, updated_task: TaskPayloadComputed) -> Task | None:
    """
    Update an existing task in DB.
    
    Args:
        db: Database session
        task_id: UUID of the task to update
        updated_task: TaskPayloadComputed with updated field values
        
    Returns:
        Updated Task ORM object, or None on failure
    """
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            logger.error(f"Task not found: {task_id}")
            return None
        
        logger.info(f"Starting task update: {task_id}")
        
        location = _ensure_location(db, updated_task.location)
        
        task.name = updated_task.name
        task.start = updated_task.start
        task.deadline = updated_task.deadline
        task.difficulty = updated_task.difficulty
        task.duration = updated_task.duration
        task.location_id = location.id
        task.importance = updated_task.importance
        task.urgency = updated_task.urgency
        task.value = updated_task.value
        task.fixed_time = updated_task.fixed_time
        task.fixed_start = updated_task.fixed_start
        
        logger.info(f"Updated task fields for: {task.name}")
        
        db.query(TaskCategory).filter(TaskCategory.task_id == task_id).delete()
        logger.debug(f"Deleted old TaskCategory records for task: {task_id}")
        
        for priority, cat_name in enumerate(updated_task.category):
            category = _ensure_category(db, cat_name)
            db.add(TaskCategory(
                task_id=task.id,
                category_id=category.id,
                priority=priority
            ))
            _ensure_category_statistics(db, cat_name)
        
        logger.info(f"Added new TaskCategory records for task: {task_id}")
        
        db.commit()
        logger.info(f"Task updated successfully: {task_id}")
        return task
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update task '{task_id}': {e}")
        return None