from pydantic import BaseModel, Field
from typing import List, Any, Optional
from datetime import datetime
import uuid


class Turn(BaseModel):
    # unique id
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str
    turn_index: int = 0

    role: str = ""              # "attacker", "defender", "audit", "judge"
    model_used: str = ""
    output_content: Optional[str] = None

    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    status: str = "SUCCESS"     # SUCCESS, FAILED, TIMEOUT

    error_report: Optional[str] = None
    execution_data: Optional[Any] = None  # raw response dump


class Trace(BaseModel):
    #unique id
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str

    behavior_goal: str = ""             # the prompt or JBB Goal
    behavior_category: str = ""         # "manipulation/charm", etc
    expected_status: str = ""           # "safe" or "unsafe"
    final_verdict: Optional[str] = None # "safe"/"unsafe"/"guarded" from the judge

    turns_used: int = 0

    total_tokens: int = 0
    total_duration_ms: float = 0.0

    status: str = "PENDING"  # SUCCESS, FAILED, RUNNING
    error_report: Optional[str] = None

    metadata: Optional[Any] = []    # extra


class Run(BaseModel):
    # unique id
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])

    timestamp: datetime = Field(default_factory=datetime.now)

    benchmark_mode: str = ""            # "defender_set", "attacker_jbb", "attacker_vs_defender"

    dataset_path: str = ""
    defender_module: str = ""
    attacker_module: str = ""
    judge_module: str = ""

    defender_model: str = ""
    attacker_model: str = ""
    judge_model: str = ""

    backend: str = ""                   # "ollama" or "vllm"

    total_traces: int = 0
    asr: Optional[float] = None         # for jbb
    refusal_rate: Optional[float] = None
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    tp_count: int = 0
    tn_count: int = 0
    fp_count: int = 0
    fn_count: int = 0

    avg_tokens_per_trace: float = 0.0
    avg_duration_ms_per_trace: float = 0.0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    config: Optional[Any] = None        # full config snapshot