from pydantic import BaseModel
from uuid import UUID
from typing import Optional, Literal


class MatchResult(BaseModel):
    """
    Output schema for TaskMatcher.find_match()
    
    Used to determine how to handle task enrichment:
    - "same": Use existing task's statistics directly
    - "similar": Aggregate from similar tasks
    - "none": Use cold-start defaults
    """
    associated_id: Optional[UUID] = None
    association_status: Literal["same", "similar", "none"]
    name_vector: list[float]  # 384-dim embedding from MiniLM