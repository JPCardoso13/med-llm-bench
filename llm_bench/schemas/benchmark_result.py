from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field

class BenchmarkResult(BaseModel):
    sample_id: str
    dataset: str
    task_name: str
    sample_type: str

    prompt: str
    response: str

    total_latency_ms: float
    ttft_ms: float
    input_tokens: int
    output_tokens: int
    inter_token_latencies_ms: List[float] = Field(default_factory=list)

    backend_metrics: Dict[str, Any] = Field(default_factory=dict)

    ref_fields: Dict[str, Any] = Field(default_factory=dict)

    cognitive_scores: Dict[str, float] = Field(default_factory=dict)

    model_id: str
    backend: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
