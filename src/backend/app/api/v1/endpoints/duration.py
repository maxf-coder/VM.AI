from fastapi import APIRouter

from app.schemas.duration import DurationPredictRequest, DurationPredictResponse
from app.services.duration import duration_service

router = APIRouter()


@router.post("/predict-duration", response_model=DurationPredictResponse)
def predict_duration(payload: DurationPredictRequest):
    result = duration_service.predict(**payload.model_dump())
    return DurationPredictResponse(predicted_duration=result)
