from pydantic import BaseModel, Field
from typing import Any, Dict, List
from datetime import datetime

class BenchmarkResult(BaseModel):
    # Identity
    sample_id: str
    dataset: str
    task_name: str
    sample_type: str  # "mcq", "generative"

    # Input and output
    prompt: str
    response: str

    # Raw measurements
    total_latency_ms: float
    ttft_ms: float
    input_tokens: int
    output_tokens: int
    inter_token_latencies_ms: List[float] = Field(default_factory=list)

    # Supplementary backend-provided timing data
    backend_metrics: Dict[str, Any] = Field(default_factory=dict)

    # Reference data for cognitive evaluation (e.g., ref_reasoning)
    ref_fields: Dict[str, Any] = Field(default_factory=dict)

    # Cognitive evaluation output (populated after inference)
    cognitive_scores: Dict[str, float] = Field(default_factory=dict)

    # Run provenance
    model_id: str
    backend: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
