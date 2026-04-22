from abc import ABC, abstractmethod
from typing import List, Dict
from llm_bench.schemas import BenchmarkResult


class BaseBackend(ABC):

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], sample_id: str, dataset: str,
                 task_name: str, sample_type: str,
                 ref_fields: dict, grouping: dict) -> BenchmarkResult:
        """
        Send a prompt to the model and return a populated BenchmarkResult.
        Timing measurements must be captured here, at the point of generation.
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Label identifying this backend, e.g. 'vllm', 'openai'."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """The model identifier as known to the serving backend."""
        pass