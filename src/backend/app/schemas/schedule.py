from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from datetime import datetime, date
from uuid import UUID
from app.schemas.shared import SuccessResponse
from app.schemas.task import TaskDetailResponse


def naive_datetime_serializer(dt: datetime) -> str:
    """Serialize datetime as naive ISO string (no timezone)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


class SchedulingResult(BaseModel):
    """Result of scheduling a single task."""
    
    model_config = ConfigDict(
        json_encoders={datetime: naive_datetime_serializer}
    )
    
    success: bool
    task_id: Optional[UUID] = None
    slot_id: Optional[UUID] = None
    slot_start: Optional[datetime] = Field(
        None,
        json_schema_extra={"example": "2026-05-20T09:00:00"}
    )
    slot_end: Optional[datetime] = Field(
        None,
        json_schema_extra={"example": "2026-05-20T10:30:00"}
    )
    displaced_tasks: List[UUID] = []
    message: str = ""


class BatchSchedulingResult(BaseModel):
    """Internal result of batch scheduling (service layer)."""
    
    model_config = ConfigDict(
        json_encoders={datetime: naive_datetime_serializer}
    )
    
    scheduled_count: int
    failed_count: int
    unscheduled_remaining: List[UUID] = []
    results: List[SchedulingResult] = []
    execution_time_ms: int = 0


class BatchScheduleResponse(BaseModel):
    """Response for POST /schedule/batch"""
    
    model_config = ConfigDict(
        json_encoders={datetime: naive_datetime_serializer}
    )
    
    success: bool
    scheduled_count: int
    failed_count: int
    unscheduled_remaining: List[UUID]
    results: List[SchedulingResult]
    execution_time_ms: int


class ScheduleTask(BaseModel):
    """A single task shown on the calendar for a specific day."""
    
    model_config = ConfigDict(
        json_encoders={datetime: naive_datetime_serializer}
    )
    
    task_id: UUID
    name: str
    start: datetime = Field(json_schema_extra={"example": "2026-05-20T09:00:00"})
    end: datetime = Field(json_schema_extra={"example": "2026-05-20T10:30:00"})
    location: str  # Not optional
    rated: bool


class ScheduleResponse(BaseModel):
    """Response for GET /schedule"""
    date: date  # Strict date type
    tasks: List[ScheduleTask]


# ---------------------------------------------------------
# Provisional Schemas
# ---------------------------------------------------------

class ProvisionalChange(BaseModel):
    """A single pending change in the schedule."""
    
    model_config = ConfigDict(
        json_encoders={datetime: naive_datetime_serializer}
    )
    
    provisional_schedule_slot_id: UUID
    task_id: UUID
    task_name: str
    change_type: str
    new_slot_start: datetime = Field(json_schema_extra={"example": "2026-05-20T09:00:00"})
    new_slot_end: datetime = Field(json_schema_extra={"example": "2026-05-20T10:30:00"})
    location: str


class ProvisionalChangesResponse(BaseModel):
    """Response for GET /provisional/changes"""
    changes: List[ProvisionalChange]
    total_count: int


class ProvisionalResetResponse(SuccessResponse):
    """Response for POST /provisional/reset"""
    changes_discarded: int


class ProvisionalCommitResponse(SuccessResponse):
    """Response for POST /provisional/commit"""
    committed_count: int
    transaction_time_ms: int
