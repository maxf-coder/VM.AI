from pydantic import BaseModel, model_validator, Field, ConfigDict
from typing import Optional, List, Any
from uuid import UUID
from datetime import datetime
from app.schemas.shared import SuccessResponse


def naive_datetime_serializer(dt: datetime) -> str:
    """Serialize datetime as naive ISO string (no timezone)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


# 1. The input structure for creating/updating tasks
class TaskPayload(BaseModel):
    """
    Clean task data with strict validation constraints.
    
    All datetime fields use naive format (no timezone).
    """
    
    model_config = ConfigDict(
        json_encoders={datetime: naive_datetime_serializer}
    )

    name: str = Field(..., min_length=1, description="Task name cannot be empty")

    start: Optional[datetime] = Field(
        None,
        json_schema_extra={"example": "2026-05-09T09:30:00"}
    )
    deadline: Optional[datetime] = Field(
        None,
        json_schema_extra={"example": "2026-05-14T17:00:00"}
    )

    difficulty: float = Field(
        ..., gt=0.0, le=1.0, description="Must be between 0.0 and 1.0"
    )
    duration: int = Field(
        ..., gt=0, lt=1440, description="Must be positive minutes < 1440"
    )

    category: List[str] = Field(
        ..., min_length=1, description="At least one category required"
    )
    location: str  # Not optional

    importance: float = Field(
        ..., gt=0.0, le=1.0, description="Must be between 0.0 and 1.0"
    )

    fixed_time: bool = False
    fixed_start: Optional[datetime] = Field(
        None,
        json_schema_extra={"example": "2026-05-09T09:30:00"}
    )

    @model_validator(mode="after")
    def check_fixed_logic(self):
        """
        Validates the relationship between fixed_time and temporal fields.
        """
        if self.fixed_time:
            if self.start is not None or self.deadline is not None:
                raise ValueError(
                    "If fixed_time is true, start and deadline must be null."
                )
            if self.fixed_start is None:
                raise ValueError("If fixed_time is true, fixed_start is required.")
        else:
            if self.start is None or self.deadline is None:
                raise ValueError(
                    "If fixed_time is false, start and deadline are required."
                )
            if self.fixed_start is not None:
                raise ValueError("If fixed_time is false, fixed_start must be null.")
        
        return self
    
    @model_validator(mode="after")
    def check_datetime_validity(self):
        """
        Validates that deadline is in the future and start < deadline.
        Uses naive datetime (no timezone conversion).
        """
        now = datetime.now()
        
        if self.start is not None and self.deadline is not None:
            if self.start >= self.deadline:
                raise ValueError("start must be before deadline")
        
        if self.deadline is not None:
            if self.deadline <= now:
                raise ValueError("deadline must be in the future")
        
        if self.fixed_time and self.fixed_start is not None:
            if self.fixed_start <= now:
                raise ValueError("fixed_start must be in the future")
        
        return self


class InternalTaskPayload(BaseModel):
    """
    Task payload for internal responses.
    
    Same as TaskPayload but without deadline/fixed_start future validation.
    Used for reading tasks that may have past deadlines.
    Uses naive datetime format (no timezone).
    """
    
    model_config = ConfigDict(
        json_encoders={datetime: naive_datetime_serializer}
    )

    name: str = Field(..., min_length=1)
    start: Optional[datetime] = Field(
        None,
        json_schema_extra={"example": "2026-05-09T09:30:00"}
    )
    deadline: Optional[datetime] = Field(
        None,
        json_schema_extra={"example": "2026-05-14T17:00:00"}
    )
    difficulty: float = Field(..., gt=0.0, le=1.0)
    duration: int = Field(..., gt=0, lt=1440)
    category: List[str] = Field(..., min_length=1)
    location: str
    importance: float = Field(..., gt=0.0, le=1.0)
    fixed_time: bool = False
    fixed_start: Optional[datetime] = Field(
        None,
        json_schema_extra={"example": "2026-05-09T09:30:00"}
    )

    @model_validator(mode="after")
    def check_fixed_logic(self):
        """Same fixed_time validation as TaskPayload."""
        if self.fixed_time:
            if self.start is not None or self.deadline is not None:
                raise ValueError("If fixed_time is true, start and deadline must be null.")
            if self.fixed_start is None:
                raise ValueError("If fixed_time is true, fixed_start is required.")
        else:
            if self.start is None or self.deadline is None:
                raise ValueError("If fixed_time is false, start and deadline are required.")
            if self.fixed_start is not None:
                raise ValueError("If fixed_time is false, fixed_start must be null.")
        return self

    @model_validator(mode="after")
    def check_datetime_validity(self):
        """Only validates start < deadline (NOT deadline > now or fixed_start > now).
        Uses naive datetime (no timezone conversion).
        """
        if self.start is not None and self.deadline is not None:
            if self.start >= self.deadline:
                raise ValueError("start must be before deadline")
        return self


# 2. Request Wrappers
class TaskCreateRequest(BaseModel):
    """Input for POST /tasks (Commit Phase)"""

    draft_id: Optional[UUID] = (
        None  # Optional: Only required if committing from Chat/AI
    )
    task: TaskPayload


class TaskUpdateRequest(BaseModel):
    """Input for POST /tasks/{id}/update"""

    task: TaskPayload


class ParseAddRequest(BaseModel):
    """Input for POST /tasks/parse/add"""

    prompt: str


class ParseModifyRequest(BaseModel):
    """Input for POST /tasks/parse/modify"""

    task: TaskPayload
    prompt: str


# 3. Response Wrappers
class TaskResponse(SuccessResponse):
    """Response for POST /tasks and POST /tasks/{id}/update"""

    task_id: UUID
    status: str = "unscheduled"


class ParseAddResponse(BaseModel):
    """Response for POST /tasks/parse/add"""

    draft_id: UUID
    task: TaskPayload


class ParseModifyResponse(BaseModel):
    """Response for POST /tasks/parse/modify"""

    task: TaskPayload


# 4. Task Detail (Read Model)
class TaskDetailResponse(BaseModel):
    """Detailed task data returned when fetching a single task or in queues."""
    
    model_config = ConfigDict(
        json_encoders={datetime: naive_datetime_serializer}
    )

    task_id: UUID
    task: InternalTaskPayload
    created_at: datetime = Field(
        json_schema_extra={"example": "2026-05-03T12:00:00"}
    )


# 5. Unscheduled Queue Schema
class UnscheduledResponse(BaseModel):
    """Response for GET /tasks/unscheduled"""
    
    tasks: List[TaskDetailResponse]
    total_count: int
