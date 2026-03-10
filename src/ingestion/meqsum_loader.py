import logging
from typing import List, Optional

from pydantic import ValidationError

from .base_loader import BaseLoader, BenchmarkSample
from src.schemas import GenerativeSample

logger = logging.getLogger(__name__)


class MeQSumLoader(BaseLoader):
    """
    Loader for the MeQSum dataset.

    Attributes:
        path_or_name (str): Local file path or dataset identifier.
                            Defaults to 'bigbio/meqsum'.
        task_type (str): Task format to load. Must be "generation" for this loader.
        split (Optional[str]): Dataset split name if applicable. Defaults to "train".
    """

    def __init__(
        self,
        path_or_name: str = "bigbio/meqsum",
        task_type: str = "generation",
        split: Optional[str] = "train",
    ):
        super().__init__(path_or_name, task_type, split)

        if self.task_type != "generation":
            raise ValueError("MeQSumLoader only supports task_type='generation'.")

    def load(self) -> List[BenchmarkSample]:
        print(
            f"Loading MeQSum from {self.path_or_name} "
            f"(Task: {self.task_type}, Split: {self.split})..."
        )

        if self._is_local_source():
            raw_data = self._read_jsonl()
        else:
            from datasets import load_dataset

            raw_data = load_dataset(self.path_or_name, "meqsum_source", split=self.split)

        samples: List[BenchmarkSample] = []
        print(f"Processing {len(raw_data)} entries...")

        invalid_count = 0
        for idx, entry in enumerate(raw_data):
            try:
                samples.append(
                    GenerativeSample(
                        id=str(entry.get("id")),
                        question=entry.get("text_1"),
                        answer=entry.get("text_2"),
                        source="meqsum",
                    )
                )
            except ValidationError as e:
                invalid_count += 1
                logger.warning("Invalid entry at index %s: %s", idx, e)
                continue

        if invalid_count:
            logger.warning("Skipped %s invalid entries during MeQSum load.", invalid_count)

        print(f"Loaded {len(samples)}/{len(raw_data)} MeQSum samples.")
        return samples


if __name__ == "__main__":
    loader = MeQSumLoader()
    data = loader.load()
    if data:
        print(f"\nFirst Sample: {data[0]}")
        print(f"\nCategory: {data[0].category}")
        print(f"\nData Type: {type(data[0])}")