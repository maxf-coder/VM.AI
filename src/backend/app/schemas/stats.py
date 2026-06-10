from pydantic import BaseModel, model_validator
from uuid import UUID
from app.schemas.shared import SuccessResponse


class RateRequest(BaseModel):
    completed: bool
    actual_duration: int | None = None
    actual_difficulty: float | None = None

    @model_validator(mode='after')
    def check_completion_details(self):
        if self.completed:
            if self.actual_duration is None or self.actual_difficulty is None:
                raise ValueError("If completed is true, actual_duration and actual_difficulty are required.")
            if not (0 < self.actual_duration < 1440):
                raise ValueError("actual_duration must be between 0 and 1440.")
            if not (0.0 < self.actual_difficulty <= 1.0):
                raise ValueError("actual_difficulty must be between 0.0 and 1.0.")
        else:
            if self.actual_duration is not None or self.actual_difficulty is not None:
                raise ValueError("If completed is false, actual_duration and actual_difficulty cannot be sent.")
        return self


class RateResponse(SuccessResponse):
    task_id: UUID