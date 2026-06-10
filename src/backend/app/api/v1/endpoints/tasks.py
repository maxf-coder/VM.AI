import io

from fastapi import APIRouter, Depends, Query, status, Path, HTTPException, UploadFile, File
from PIL import Image
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
from datetime import datetime

from app.core.database import get_db
from app.core.logging_config import setup_logging
from app.models.draft import TaskDraft
from app.models.schedule import MainScheduleSlot, ProvisionalSlot
from app.models.task import Task
from app.utils import normalize_task_payload
from app.schemas.task import (
    TaskCreateRequest,
    TaskUpdateRequest,
    TaskResponse,
    TaskDetailResponse,
    TaskPayload,
    InternalTaskPayload,
    ParseAddRequest,
    ParseModifyRequest,
    ParseAddResponse,
    ParseModifyResponse,
    UnscheduledResponse,
)
from app.services.task_matcher import task_matcher
from app.services.enrichment import enrichment_service
from app.utils.task_saver import save_commited_task, update_commited_task
from app.models.workflow import UnscheduledTask
from app.services.parser import parser_service
from app.services.stats_recorder import stats_recorder
from app.services.img_to_prompt import img_to_prompt

router = APIRouter()
logger = setup_logging()


# ---------------------------------------------------------
# 1. NLP Parsing Endpoints
# ---------------------------------------------------------


@router.post("/parse/add", response_model=ParseAddResponse)
def parse_add_task(
    body: ParseAddRequest,
    db: Session = Depends(get_db),
):
    """
    POST /tasks/parse/add
    Parses natural language input to extract task fields.
    """
    try:
        logger.info(f"Parse add started: '{body.prompt}'")

        # Step 1: Parse prompt
        nlp_payload = parser_service.parse_add(body.prompt)
        if not nlp_payload:
            logger.error("Parser returned None")
            raise HTTPException(status_code=500, detail="Parser failed to parse prompt")

        logger.debug(f"Parser output: {nlp_payload.model_dump()}")

        # Step 2: Find match
        match_result = task_matcher.find_match(db, nlp_payload.name.value)
        if not match_result:
            logger.error("Task matcher returned None")
            raise HTTPException(status_code=500, detail="Task matcher failed")

        logger.debug(f"Match result: status={match_result.association_status}, id={match_result.associated_id}")

        # Step 3: Enrichment
        task_payload, draft_id = enrichment_service.predict_nlp_add(db, nlp_payload, match_result)

        if not task_payload or not draft_id:
            logger.error("Enrichment returned None")
            raise HTTPException(status_code=500, detail="Enrichment failed")

        logger.debug(f"Enrichment output: name={task_payload.name}, fixed_time={task_payload.fixed_time}")

        logger.info(f"Parse add complete. Draft ID: {draft_id}")

        return ParseAddResponse(task=task_payload, draft_id=draft_id)
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Parse add failed:")  # This prints full traceback
        raise HTTPException(status_code=500, detail=f"Parse add failed")


@router.post("/parse/modify", response_model=ParseModifyResponse)
def parse_modify_task(
    body: ParseModifyRequest,
    db: Session = Depends(get_db),
):
    """
    POST /tasks/parse/modify
    Parses modification prompts.
    """
    try:
        logger.info(f"Parse modify started: '{body.prompt}'")

        # Step 1: Parse modification
        changed_fields = parser_service.parse_modify(body.task, body.prompt)
        if not changed_fields:
            logger.error("Parser returned None")
            raise HTTPException(status_code=500, detail="Parser failed to modify task")

        logger.info(f"Parser output: {changed_fields}")

        # Step 2: Merge with existing task
        merged_task = enrichment_service.merge_nlp_modify(db, body.task, changed_fields)
        if not merged_task:
            logger.error("Merge returned None")
            raise HTTPException(status_code=500, detail="Merge failed")

        logger.debug(f"Merged task: {merged_task.model_dump()}")

        logger.info("Parse modify complete")

        return ParseModifyResponse(task=merged_task)
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Parse modify failed: {e}")
        raise HTTPException(status_code=500, detail="Parse modify failed")


@router.post("/parse/from-image")
async def parse_from_image(
    file: UploadFile = File(...),
):
    """
    POST /tasks/parse/from-image
    Upload an image → classify activity → return a task prompt string.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {file.content_type}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = img_to_prompt.classify(image)
    return {"prompt": result["prompt"]}


# ---------------------------------------------------------
# 2. Queue Endpoint
# ---------------------------------------------------------


@router.get("/unscheduled", response_model=UnscheduledResponse)
def get_unscheduled(
    limit: Optional[int] = Query(None, description="Max number of tasks to return"),
    db: Session = Depends(get_db),
):
    """
    GET /tasks/unscheduled
    Fetches tasks waiting for scheduling (FIFO order).
    """
    logger.info("Fetching unscheduled tasks")

    query = db.query(UnscheduledTask).order_by(UnscheduledTask.created_at)
    
    if limit is not None:
        query = query.limit(limit)
    
    unscheduled = query.all()

    tasks = []
    for entry in unscheduled:
        task = entry.task
        category_names = [tc.category.name for tc in task.task_categories]

        tasks.append(TaskDetailResponse(
            task_id=task.id,
            task=InternalTaskPayload(
                name=task.name,
                start=task.start,
                deadline=task.deadline,
                difficulty=task.difficulty,
                duration=task.duration,
                category=category_names,
                location=task.location.name,
                importance=task.importance,
                fixed_time=task.fixed_time,
                fixed_start=task.fixed_start,
            ),
            created_at=task.created_at,
        ))

    logger.info(f"Returning {len(tasks)} unscheduled tasks")

    return UnscheduledResponse(
        tasks=tasks,
        total_count=len(tasks),
    )


# ---------------------------------------------------------
# 3. Task CRUD Endpoints
# ---------------------------------------------------------


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=TaskResponse)
def create_task(
    body: TaskCreateRequest,
    db: Session = Depends(get_db),
):
    """
    POST /tasks
    Creates a new task in the database.

    Flow:
        If draft_id is present:
            1. Fetch draft from DB.
            2. Update draft data with body.task edits.
            3. Save to main DB, delete draft.
        Else (Manual Creation):
            1. Run Task Matching & Enrichment pipeline on body.task.
            2. Save to main DB.
    """
    try:
        logger.info("Starting task commit...")
        logger.info(f"Request body: {body.model_dump()}")
        
        normalize_task_payload(body.task)
        
        logger.debug(f"Normalized task: {body.task.model_dump()}")
        
        enriched = None

        if body.draft_id:
            logger.info(f"Request body have draft_id: {body.draft_id}")
            
            # Check if draft exists in DB
            draft = db.query(TaskDraft).filter(TaskDraft.id == body.draft_id).first()
            if not draft:
                logger.warning(f"Draft not found: {body.draft_id}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Draft with id {body.draft_id} not found"
                )
            
            enriched = enrichment_service.commit_from_draft(db, body.task, body.draft_id)
            if not enriched:
                logger.warning("No output from enrichment")
                raise HTTPException(
                    status_code=500,
                    detail="The enrichment failed for commit_from_draft"
                )
            logger.info(f"Output from commit_from_draft(): {enriched.model_dump()}")
        else:
            logger.info("Request body don't have draft_id")
            match_result = task_matcher.find_match(db, body.task.name)
            if not match_result:
                logger.warning("No output from task_matcher")
                raise HTTPException(
                    status_code=500,
                    detail="The task matching failed"
                )
            logger.info(f"Match result: status={match_result.association_status}, id={match_result.associated_id}")
            enriched = enrichment_service.commit_manual(db, body.task, match_result)
            if not enriched:
                logger.warning("No output from enrichment")
                raise HTTPException(
                    status_code=500,
                    detail="The enrichment failed for commit_manual"
                )
            logger.info(f"Committed task: name={enriched.name}, value={enriched.value}")

        saved = save_commited_task(db, enriched)

        if not saved:
            logger.error("Failed to save task to database")
            raise HTTPException(
                status_code=500,
                detail="Failed to save task to database"
            )
        
        stats_updated = stats_recorder.update_stats_after_commit(db, saved.id)

        if not stats_updated:
            logger.error("Failed to update stats")
            raise HTTPException(
                status_code=500,
                detail="Failed to update task statistics"
            )

        db.commit()
        
        logger.info(f"Task committed successfully: {saved.id}")
        
        return TaskResponse(
            success=True,
            task_id=saved.id,
            status="unscheduled",
            message="Task enrichment complete - saved to DB",
        )
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Task commit failed: {e}")
        raise HTTPException(status_code=500, detail="Task commit failed")


@router.post("/{id}/update", response_model=TaskResponse)
def update_task(
    body: TaskUpdateRequest,
    db: Session = Depends(get_db),
    id: UUID = Path(..., description="ID of the task to update"),
    source: str = Query(..., description="main_schedule | unscheduled | provisional"),
):
    """
    POST /tasks/{id}/update
    Updates an existing task based on its source.
    """
    try:
        logger.info(f"Starting task update: {id}")
        
        if source == "provisional":
            slot = db.query(ProvisionalSlot).filter(ProvisionalSlot.task_id == id).first()
            if not slot:
                raise HTTPException(status_code=400, detail="Task not in provisional schedule")
            stats_recorder.update_time_score(db, id, slot.start, boost=-2.0)
        elif source == "main_schedule":
            slot = db.query(MainScheduleSlot).filter(MainScheduleSlot.task_id == id).first()
            if not slot:
                raise HTTPException(status_code=400, detail="Task not in main schedule")
            
            now = datetime.now()
            if slot.end < now:
                raise HTTPException(status_code=400, detail="Cannot update task that ended in the past")
            
            task = db.query(Task).filter(Task.id == id).first()
            if task and task.rated:
                raise HTTPException(status_code=409, detail="Cannot update rated task")
            
            stats_recorder.update_time_score(db, id, slot.start, boost=-1.0)
        
        normalize_task_payload(body.task)
        
        computed_task = enrichment_service.update_task(db, body.task)
        
        updated = update_commited_task(db, id, computed_task)
        
        if not updated:
            logger.error(f"Failed to update task in DB: {id}")
            raise HTTPException(
                status_code=500,
                detail="Failed to update task in database"
            )
        
        stats_updated = stats_recorder.update_stats_after_commit(db, id)
        
        if not stats_updated:
            logger.error(f"Failed to update stats for task: {id}")
            raise HTTPException(
                status_code=500,
                detail="Failed to update task statistics"
            )
        
        db.query(UnscheduledTask).filter(UnscheduledTask.task_id == id).delete()
        unscheduled = UnscheduledTask(task_id=id)
        db.add(unscheduled)

        # Clean up any existing provisional slot when re-adding to unscheduled
        existing_slot = db.query(ProvisionalSlot).filter(ProvisionalSlot.task_id == id).first()
        if existing_slot:
            logger.debug(f"Deleting provisional slot for task {id} when re-adding to unscheduled")
            db.delete(existing_slot)
        
        db.commit()
        
        logger.info(f"Task updated successfully: {id}")
        
        return TaskResponse(
            success=True,
            task_id=id,
            status="unscheduled",
            message="Task updated successfully",
        )
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Task update failed: {e}")
        raise HTTPException(status_code=500, detail="Task update failed")


@router.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(
    id: UUID = Path(..., description="ID of the task to delete"),
    source: str = Query(..., description="main_schedule | unscheduled | provisional | tasks"),
    db: Session = Depends(get_db),
):
    """
    DELETE /tasks/{id}
    Deletes a task based on its source context.
    """
    try:
        logger.info(f"Starting task delete: {id}, source: {source}")
        
        if source == "main_schedule":
            slot = db.query(MainScheduleSlot).filter(MainScheduleSlot.task_id == id).first()
            if not slot:
                logger.info(f"Task not found in main_schedule: {id}")
                raise HTTPException(status_code=404, detail="Task not found in main_schedule")
            
            task = db.query(Task).filter(Task.id == id).first()
            if not task:
                logger.info(f"Task not found: {id}")
                raise HTTPException(status_code=404, detail="Task not found")
            
            db.query(Task).filter(Task.id == id).delete()
            logger.info(f"Deleted task from tasks table: {id}")
            
        elif source == "unscheduled":
            in_provisional = db.query(ProvisionalSlot).filter(ProvisionalSlot.task_id == id).first()
            in_main = db.query(MainScheduleSlot).filter(MainScheduleSlot.task_id == id).first()
            
            if in_provisional or in_main:
                record = db.query(UnscheduledTask).filter(UnscheduledTask.task_id == id).first()
                if not record:
                    logger.info(f"Task not found in unscheduled_tasks: {id}")
                    raise HTTPException(status_code=404, detail="Task not found in unscheduled queue")
                db.query(UnscheduledTask).filter(UnscheduledTask.task_id == id).delete()
                logger.info(f"Deleted task from unscheduled_tasks: {id}")
            else:
                task = db.query(Task).filter(Task.id == id).first()
                if not task:
                    logger.info(f"Task not found: {id}")
                    raise HTTPException(status_code=404, detail="Task not found")
                db.query(Task).filter(Task.id == id).delete()
                logger.info(f"Deleted task from tasks table: {id}")
                
        elif source == "provisional":
            in_unscheduled = db.query(UnscheduledTask).filter(UnscheduledTask.task_id == id).first()
            in_main = db.query(MainScheduleSlot).filter(MainScheduleSlot.task_id == id).first()
            
            if in_unscheduled or in_main:
                slot = db.query(ProvisionalSlot).filter(ProvisionalSlot.task_id == id).first()
                if not slot:
                    logger.info(f"Task not found in provisional_schedule: {id}")
                    raise HTTPException(status_code=404, detail="Task not found in provisional schedule")
                db.query(ProvisionalSlot).filter(ProvisionalSlot.task_id == id).delete()
                logger.info(f"Deleted task from provisional_schedule: {id}")
            else:
                task = db.query(Task).filter(Task.id == id).first()
                if not task:
                    logger.info(f"Task not found: {id}")
                    raise HTTPException(status_code=404, detail="Task not found")
                db.query(Task).filter(Task.id == id).delete()
                logger.info(f"Deleted task from tasks table: {id}")
                
        elif source == "tasks":
            task = db.query(Task).filter(Task.id == id).first()
            if not task:
                logger.info(f"Task not found: {id}")
                raise HTTPException(status_code=404, detail="Task not found")
            db.query(Task).filter(Task.id == id).delete()
            logger.info(f"Deleted task from tasks table: {id}")
        else:
            raise HTTPException(status_code=400, detail="Invalid source parameter")
        
        db.commit()
        logger.info(f"Task delete completed: {id}")
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Task delete failed: {e}")
        raise HTTPException(status_code=500, detail="Task delete failed")


# ---------------------------------------------------------
# 4. Task Fetching Endpoint
# ---------------------------------------------------------


@router.get("/{id}", response_model=TaskDetailResponse)
def get_task(
    id: UUID,
    db: Session = Depends(get_db),
):
    """
    GET /tasks/{id}
    Fetches details of a specific task by ID.
    """
    logger.info(f"Fetching task: {id}")

    task = db.query(Task).filter(Task.id == id).first()
    if not task:
        logger.info(f"Task not found: {id}")
        raise HTTPException(status_code=404, detail="Task not found")

    category_names = [tc.category.name for tc in task.task_categories]

    logger.info(f"Task fetched successfully: {id}")

    return TaskDetailResponse(
        task_id=task.id,
        task=InternalTaskPayload(
            name=task.name,
            start=task.start,
            deadline=task.deadline,
            difficulty=task.difficulty,
            duration=task.duration,
            category=category_names,
            location=task.location.name,
            importance=task.importance,
            fixed_time=task.fixed_time,
            fixed_start=task.fixed_start,
        ),
        created_at=task.created_at,
    )

