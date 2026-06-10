from pydantic import BaseModel, Field
from typing import Any, Optional, List

from app.schemas.task import TaskPayload


# ================================================================================
# NlpPayloadField - Single field with {value, predicted}
# ================================================================================

class NlpPayloadField(BaseModel):
    """
    Single field with value and predicted flag.
    Value can be: str, list[str], int, float, datetime, None
    """

    value: Any = None
    predicted: bool = True


# ================================================================================
# NlpAddPayload - For predict_nlp_add() input
# ================================================================================

class NlpAddPayload(BaseModel):
    """
    NLP payload structure for predict_nlp_add().
    Contains all fields with {value, predicted} wrapper.
    """

    name: NlpPayloadField
    start: NlpPayloadField
    deadline: NlpPayloadField
    difficulty: NlpPayloadField
    duration: NlpPayloadField
    category: NlpPayloadField
    location: NlpPayloadField
    importance: NlpPayloadField
    fixed_time: NlpPayloadField
    fixed_start: NlpPayloadField