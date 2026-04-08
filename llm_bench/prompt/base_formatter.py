from abc import ABC, abstractmethod
from typing import List, Optional
from llm_bench.schemas.mcq_sample import MCQSample
from llm_bench.schemas.generative_sample import GenerativeSample

Sample = MCQSample | GenerativeSample
class BaseFormatter(ABC):

    @abstractmethod
    def format(
        self,
        sample: Sample,
        fewshot_examples: Optional[List[Sample]] = None,
    ) -> str:
        """
        Format a sample into a prompt string ready to send to the model.
        Fewshot examples are optional and handled here if provided.
        """
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The system prompt for this task."""
        pass