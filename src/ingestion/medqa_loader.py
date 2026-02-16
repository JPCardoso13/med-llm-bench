import logging
import uuid
from typing import List, Optional
from src.ingestion.base_loader import BaseLoader, BenchmarkSample
from src.schemas.mcqsample import MCQSample
from src.schemas.generativesample import GenerativeSample
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class MedQALoader(BaseLoader):
    """
    Loader for the MedQA dataset.

    Attributes:
        path_or_name (str): Local file path or dataset identifier.
                            Defaults to 'GBaker/MedQA-USMLE-4-options'.
        task_type (str): Task format to load, either "mcq" or "generation".
        split (Optional[str]): Dataset split name if applicable. Defaults to "test".

    Raises:
        ValueError: If task_type is not one of the supported values.
    """
    def __init__(self, 
                 path_or_name: str = "GBaker/MedQA-USMLE-4-options", 
                 task_type: str = "mcq", 
                 split: Optional[str] = "test"):
        super().__init__(path_or_name, task_type, split)

    def load(self) -> List[BenchmarkSample]:
        print(f"Loading MedQA from {self.path_or_name} (Task: {self.task_type})...")

        if self._is_local_source():
            raw_data = self._read_jsonl()
        else:
            from datasets import load_dataset
            raw_data = load_dataset(self.path_or_name, split=self.split)

        samples = []
        
        print(f"Processing {len(raw_data)} entries...")

        invalid_count = 0
        for idx, entry in enumerate(raw_data): # NOTE: idx is the literal index of the 'raw_data' list,
                                               # which may differ from the original dataset's 'id' field if it exists.
            # ID is absent in the raw dataset, so we generate a unique one for each sample.
            unique_id = f"medqa_{str(uuid.uuid4())[:8]}"
            
            if self.task_type == "mcq":
                try:
                    samples.append(MCQSample(
                        id=unique_id,
                        question=entry.get('question'),
                        options=entry.get("options"),
                        answer_idx=entry.get("answer_idx"),
                        source="medqa",
                        category=entry.get('meta_info')
                    ))
                except ValidationError as e:
                    invalid_count += 1
                    logger.warning("Invalid entry at index %s: %s", idx, e)
                    continue

            elif self.task_type == "generation":
                try:
                    samples.append(GenerativeSample(
                        id=unique_id,
                        question=entry.get('question'),
                        ref_answer=entry.get('answer'), # Same text as for the right option in the MCQ format
                        source="medqa",
                        category=entry.get('meta_info')
                    ))
                except ValidationError as e:
                    invalid_count += 1
                    logger.warning("Invalid entry at index %s: %s", idx, e)
                    continue

        if invalid_count:
            logger.warning("Skipped %s invalid entries during MedQA load.", invalid_count)
        print(f"Loaded {len(samples)}/{len(raw_data)} MedQA samples.")
        return samples

if __name__ == "__main__":
    loader = MedQALoader()
    data = loader.load()
    if data:
        print(f"\nFirst Sample: {data[0]}")
        print(f"\nCategory: {data[0].category}")
        print(f"\nData Type: {type(data[0])}")
