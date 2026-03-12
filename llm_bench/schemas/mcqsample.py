from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


class MCQSample(BaseModel):
    """
    A unified representation of a multiple choice question for benchmarking.

    Used for tasks where the model must select one answer from provided options.

    Attributes:
        id (str): A unique identifier for the sample.
        question (str): The question the model should answer.
        options (Dict[str, str]): A dictionary mapping option keys to option text.
                                  Example: {"A": "Option 1", "B": "Option 2"}
        answer_idx (str): The key corresponding to the correct option.
        context (Optional[str]): Reference text if the task is context-dependent (e.g., a reading comprehension task).
        source (Optional[str]): The origin dataset name.
        grouping (Dict[str, List[str]]): Group-by axes for aggregated metrics.
                                         Example: {"specialty": ["cardiology", "oncology"], "topic": ["quantum physics"]}
        metadata (Dict[str, str]): Auxiliary metadata not intended for aggregation.
                                   Example: {"pmc_id": "PMC12345", "source_url": "https://..."}
    """

    id: str
    question: str
    options: Dict[str, str]
    answer_idx: str
    context: Optional[str] = None
    source: Optional[str] = None
    grouping: Dict[str, List[str]] = Field(default_factory=dict)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @field_validator('id', 'question', 'answer_idx')
    @classmethod
    def check_non_empty_string(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace.")
        return v

    @field_validator('options')
    @classmethod
    def check_options_length(cls, v: Dict[str, str]) -> Dict[str, str]:
        if len(v) < 2:
            raise ValueError("The 'options' dictionary must contain at least two entries.")
        return v

    @model_validator(mode='after')
    def check_answer_in_options(self) -> 'MCQSample':
        if self.answer_idx not in self.options:
            raise ValueError(f"Field 'answer_idx' ('{self.answer_idx}') must be in the 'options' dictionary.")
        return self
