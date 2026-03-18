from abc import ABC, abstractmethod
from typing import Dict, List, Any
from llm_bench.schemas import SCHEMA_REGISTRY

class BaseLoader(ABC):
    """
    Abstract base class for custom dataset loaders.
    """
    def __init__(self, schema_type: str):
        if schema_type not in SCHEMA_REGISTRY:
            valid_schemas = ", ".join(SCHEMA_REGISTRY.keys())
            raise ValueError(f"Unknown schema: '{schema_type}'. Valid options are: {valid_schemas}")
            
        self.schema_type = schema_type
        self.schema_class = SCHEMA_REGISTRY[schema_type]

    @abstractmethod
    def load(self) -> Dict[str, List[Any]]:
        """
        Load the dataset and return validated sample objects split by usage.

        Returns:
            Dict[str, List[BenchmarkSample]]: A dictionary with keys like 'eval' and 'fewshot'.
        """
        pass
