from pydantic import BaseModel, Field
from typing import Optional, Literal

# response model schema
class ResponseSchema(BaseModel):
    thought_process: str = Field(description="The chain of thought...")
    output_content: str = Field(description="The actual response...")

# audit model schema
class AuditSchema(BaseModel):
    safety_status: Literal["safe", "unsafe"] = Field(description="Safety check.")
    message_tldr: Optional[str] = Field(...)
    suggested_fix: Optional[str] = Field(...)