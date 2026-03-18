from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Set, Union

from llm_bench.schemas import MCQSample, GenerativeSample


BenchmarkSample = Union[MCQSample, GenerativeSample]


class BaseLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    """
    ALLOWED_TASK_TYPES: ClassVar[Set[str]] = {"mcq", "generation"}

    def __init__(self, task_type: str):
        self.task_type = task_type
        self._validate_task_type()

    def _validate_task_type(self) -> None:
        if self.task_type not in self.ALLOWED_TASK_TYPES:
            allowed = ", ".join(sorted(self.ALLOWED_TASK_TYPES))
            raise ValueError(
                f"{self.__class__.__name__} supports task_type in [{allowed}], "
                f"got '{self.task_type}'."
            )

    @abstractmethod
    def load(self) -> Dict[str, List[BenchmarkSample]]:
        """
        Load the dataset and return validated sample objects split by usage.

        Returns:
            Dict[str, List[BenchmarkSample]]: A dictionary with keys like 'eval' and 'fewshot'.
        """
        pass
