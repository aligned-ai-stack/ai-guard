from pydantic import BaseModel, Field
from typing import List, Any, Optional
from datetime import datetime
import uuid

class Trace(BaseModel):
    #unique id
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    #id based on the test run (so we can pair tests together)
    run_id: str

    timestamp: datetime = Field(default_factory=datetime.now)

    #context
    input_query: str
    expected_status: Optional[str] = None   #e.g.: safe, unsafe
    input_tokens: int = 0

    framework_version: str
    gen_model: str
    audit_model: str

    #execution data should be JSON, this will be compatible
    #for multiple frameworks
    execution_data: Optional[Any] = None
    predicted_status: Optional[str] = None  #e.g.: safe, unsafe
    output_content: Optional[str] = None

    status: str = "PENDING"  # SUCCESS, FAILED, RUNNING
    error_report: Optional[str] = None

    #performance
    total_duration_ms: float = 0.0
    total_tokens: int = 0

    #metadata
    tags: List[str] = []