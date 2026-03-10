from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union, Optional
import json
from src.schemas import MCQSample, GenerativeSample

BenchmarkSample = Union[MCQSample, GenerativeSample]

class BaseLoader(ABC):
    """
    Abstract base class for all dataset loaders.

    Attributes:
        path_or_name (str): Local file path or dataset identifier.
        task_type (str): Task format to load, either "mcq" or "generation".
        split (Optional[str]): Dataset split name if applicable. Defaults to None.

    Raises:
        ValueError: If task_type is not one of the supported values.
    """

    def __init__(self, path_or_name: str, task_type: str, split: Optional[str] = None):
        self.path_or_name = path_or_name
        self.task_type = task_type
        self.split = split

        if self.task_type not in ["mcq", "generation"]:
            raise ValueError(f"Invalid task_type: {self.task_type}. Must be 'mcq' or 'generation'.")

    @abstractmethod
    def load(self) -> List[BenchmarkSample]:
        """
        Load the dataset and return validated sample objects.

        Returns:
            List[BenchmarkSample]: Validated MCQ or generative samples.
        """
        pass

    def _read_jsonl(self) -> List[dict]:
        """Helper to read JSONL files."""
        data = []
        try:
            with open(self.path_or_name, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.path_or_name}")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.path_or_name}: {e}")

        return data

    def _is_local_source(self) -> bool:
        """Return True when path_or_name points to a local file."""
        return Path(self.path_or_name).is_file()
