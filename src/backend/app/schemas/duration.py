from pydantic import BaseModel


class DurationPredictRequest(BaseModel):
    difficulty: float
    importance: float
    scheduled_duration: int
    category: str
    location: str
    fixed_time: str = ""
    time_difference: float = -1


class DurationPredictResponse(BaseModel):
    predicted_duration: int
