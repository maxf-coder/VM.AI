from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from uuid import UUID

from app.schemas.task import TaskPayload


# ================================================================================
# TaskPayloadComputed - Base + Computed fields (urgency, value)
# ================================================================================

class TaskPayloadComputed(TaskPayload):
    """
    Task data with computed fields (urgency, value).
    Used as output for update_task().
    """

    urgency: float = Field(0.0, ge=0.0, le=1.0, description="Computed urgency 0.0-1.0")
    value: float = Field(0.0, ge=0.0, le=1.0, description="Computed value 0.0-1.0")


# ================================================================================
# TaskPayloadComputedWithRefs - Base + Computed + Internal References
# ================================================================================

class TaskPayloadComputedWithRefs(TaskPayloadComputed):
    """
    Complete task data with computed fields and internal references.
    Used as output for commit_from_draft() and commit_manual().
    Includes:
    - Base fields from TaskPayload
    - Computed fields: urgency, value
    - Internal references: task_statistics_id, name_vector, association_status
    """

    task_statistics_id: Optional[UUID] = None
    name_vector: Optional[List[float]] = None
    association_status: Optional[Literal["same", "similar", "none"]] = None