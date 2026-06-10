from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID
from datetime import datetime

from app.core.database import get_db
from app.core.logging_config import setup_logging
from app.models.schedule import MainScheduleSlot
from app.models.task import Task
from app.schemas.stats import RateRequest, RateResponse
from app.services.stats_recorder import stats_recorder

router = APIRouter()
logger = setup_logging()


@router.post("/{id}/rate", status_code=status.HTTP_200_OK, response_model=RateResponse)
def rate_task(
    id: UUID,
    body: RateRequest,
    db: Session = Depends(get_db),
):
    """
    POST /tasks/{id}/rate
    Rate a task as completed or uncompleted.
    """
    logger.info(f"Starting task rating: {id}, completed={body.completed}")

    slot = db.query(MainScheduleSlot).filter(MainScheduleSlot.task_id == id).first()
    if not slot:
        logger.info(f"Task not in main_schedule: {id}")
        raise HTTPException(status_code=400, detail="Task not in main schedule")

    task = db.query(Task).filter(Task.id == id).first()
    if not task:
        logger.info(f"Task not found: {id}")
        raise HTTPException(status_code=404, detail="Task not found")

    if task.rated:
        logger.info(f"Task already rated: {id}")
        raise HTTPException(status_code=400, detail="Task already rated")

    success = stats_recorder.rate_task(
        db,
        id,
        slot.start,
        body.completed,
        body.actual_duration,
        body.actual_difficulty,
    )

    if not success:
        logger.error(f"Failed to rate task: {id}")
        raise HTTPException(status_code=500, detail="Failed to rate task")

    logger.info(f"Task rated successfully: {id}")
    return RateResponse(
        success=True,
        task_id=id,
        message="Task rated successfully",
    )
