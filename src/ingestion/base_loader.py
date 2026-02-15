from abc import ABC, abstractmethod
from typing import List, Union
import json
from src.schemas.mcqsample import MCQSample
from src.schemas.generativesample import GenerativeSample

BenchmarkSample = Union[MCQSample, GenerativeSample]

class BaseLoader(ABC):
    """
    Abstract Base Class for all data loaders.
    
    Standardizes:
    1. Input arguments (dataset path + task type)
    2. Output format (List of Samples)
    """
    
    def __init__(self, file_path: str, task_type: str):
        self.file_path = file_path
        self.task_type = task_type
        
        # Enforce valid task types globally
        if self.task_type not in ["mcq", "generation"]:
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be 'mcq' or 'generation'.")

    @abstractmethod
    def load(self) -> List[BenchmarkSample]:
        """
        Must return a list of validated sample objects.
        """
        pass

    def _read_jsonl(self) -> List[dict]:
        """Helper to read JSONL files."""
        data = []
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.file_path}: {e}")

        return data
