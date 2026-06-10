from pydantic import BaseModel
from typing import Optional

class SuccessResponse(BaseModel):
    success: bool = True
    message: Optional[str] = None

