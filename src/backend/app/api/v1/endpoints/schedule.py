from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime, date, timedelta

from app.core.database import get_db
from app.core.logging_config import setup_logging
from app.models.schedule import MainScheduleSlot
from app.models.task import Task
from app.schemas.schedule import ScheduleResponse, BatchScheduleResponse, ScheduleTask
from app.services.schedule_engine import schedule_engine
from app.services.stats_recorder import stats_recorder
from app.models.workflow import UnscheduledTask

logger = setup_logging()
router = APIRouter()


@router.get("/", response_model=ScheduleResponse)
def get_schedule(
    date: date = Query(..., description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
):
    """
    GET /schedule?date=YYYY-MM-DD
    
    Returns all tasks from main_schedule that start on the specified date.
    """
    logger.debug(f"Fetching tasks for date: {date.strftime('%Y-%m-%d')}")
    day_start = datetime.combine(date, datetime.min.time())
    day_end = day_start + timedelta(days=1)
    
    slots = db.query(MainScheduleSlot).filter(
        MainScheduleSlot.start >= day_start,
        MainScheduleSlot.start < day_end,
    ).join(Task).order_by(MainScheduleSlot.start).all()
    
    tasks = []
    for slot in slots:
        tasks.append(ScheduleTask(
            task_id=slot.task_id,
            name=slot.task.name,
            start=slot.start,
            end=slot.end,
            location=slot.location or "",
            rated=slot.task.rated,
        ))
    
    return ScheduleResponse(
        date=date,
        tasks=tasks,
    )


@router.post("/batch", status_code=status.HTTP_200_OK, response_model=BatchScheduleResponse)
def schedule_batch(
    db: Session = Depends(get_db),
):
    """
    POST /schedule/batch
    """
    try:
        unscheduled_before = {entry.task_id for entry in db.query(UnscheduledTask).all()}
        
        result = schedule_engine.schedule_batch(db)
        
        for scheduling_result in result.results:
            if scheduling_result.success and scheduling_result.task_id:
                if scheduling_result.task_id in unscheduled_before:
                    if scheduling_result.slot_start:
                        stats_recorder.update_time_score(
                            db,
                            scheduling_result.task_id,
                            scheduling_result.slot_start,
                            boost=1.0
                        )
                    else:
                        logger.error(f"slot_start is None for task {scheduling_result.task_id}")
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="slot_start is None"
                        )
        
        return BatchScheduleResponse(
            success=result.scheduled_count > 0,
            scheduled_count=result.scheduled_count,
            failed_count=result.failed_count,
            unscheduled_remaining=result.unscheduled_remaining,
            results=result.results,
            execution_time_ms=result.execution_time_ms,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch scheduling failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch scheduling failed"
        )