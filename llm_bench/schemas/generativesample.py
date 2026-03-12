from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class GenerativeSample(BaseModel):
    """
    A unified representation of a generative question for benchmarking.

    Used for tasks where the model must generate a free-text response.

    Attributes:
        id (str): A unique identifier for the sample.
        question (str): The question the model should answer.
        answer (str): The reference answer (ground truth) used for evaluation.
        ref_reasoning (Optional[str]): Optional reference reasoning/explanation for the answer.
        context (Optional[str]): Reference text if the task is context-dependent (e.g., a reading comprehension task).
        source (Optional[str]): The origin dataset name.
        grouping (Dict[str, List[str]]): Group-by axes for aggregated metrics.
                                         Example: {"specialty": ["cardiology", "oncology"], "topic": ["quantum physics"]}
        metadata (Dict[str, str]): Auxiliary metadata not intended for aggregation.
                                   Example: {"pmc_id": "PMC12345", "source_url": "https://..."}
    """

    id: str
    question: str
    answer: str
    ref_reasoning: Optional[str] = None
    context: Optional[str] = None
    source: Optional[str] = None
    grouping: Dict[str, List[str]] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator('id', 'question', 'answer')
    @classmethod
    def check_non_empty_string(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace.")
        return v
