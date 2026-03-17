from pydantic import BaseModel
from typing import List, Any
from datetime import datetime

class LLMCall(BaseModel):
    model: str
    duration_ms: float
    prompt_tokens: int
    completion_tokens: int
    output: Any