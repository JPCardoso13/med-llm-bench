import logging
import uuid
from typing import List, Optional
from .base_loader import BaseLoader, BenchmarkSample
from src.schemas import MCQSample
from pydantic import ValidationError
import re

logger = logging.getLogger(__name__)

class MedXpertLoader(BaseLoader):
    """
    Loader for the MedXpertQA dataset. Only loads the 'Text' subset by defult.

    Attributes:
        path_or_name (str): Local file path or dataset identifier.
                            Defaults to 'TsinghuaC3I/MedXpertQA'.
        task_type (str): Task format to load, either "mcq" or "generation".
                         Defaults to "mcq".
        split (Optional[str]): Dataset split name if applicable. Defaults to "test".

    Raises:
        ValueError: If task_type is not one of the supported values.
    """
    def __init__(self, 
                 path_or_name: str = "TsinghuaC3I/MedXpertQA", 
                 task_type: str = "mcq", 
                 split: Optional[str] = "test"):
        super().__init__(path_or_name, task_type, split)

    def _clean_question(self, text: str) -> str:
        """
        Removes explicit answer choices from the question text 
        so we don't duplicate options in the final prompt.
        """
        pattern = r"(?i)(Answer Choices:|Options:|Select one:).*$"
        cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return cleaned

    def load(self) -> List[BenchmarkSample]:
        print(f"Loading MedXpertQA from {self.path_or_name} (Task: {self.task_type})...")

        if self._is_local_source():
            raw_data = self._read_jsonl()
        else:
            from datasets import load_dataset
            raw_data = load_dataset(self.path_or_name, "Text", split=self.split)

        samples = []
        
        print(f"Processing {len(raw_data)} entries...")
      
        invalid_count = 0
        for idx, entry in enumerate(raw_data): # NOTE: idx is the literal index of the 'raw_data' list,
                                               # which may differ from the original dataset's 'id' field if it exists.
            # Can use parameter 'id' instead of generating a new one, if needed
            unique_id = f"medxpertqa_{str(uuid.uuid4())[:8]}"
            
            question_text = entry.get('question')
            options = entry.get('options')
            answer_key = entry.get('label')
            category = entry.get('body_system') # e.g. "Cardiovascular" (Could be converted to a medical specialty)
            tags = {
                "medical_task": entry.get("medical_task"),
                "question_type": entry.get("question_type")
            }

            if self.task_type == "mcq":
                try:
                    samples.append(MCQSample(
                        id=unique_id,
                        question=self._clean_question(question_text),
                        options=options,
                        answer_idx=answer_key,
                        source="medxpertqa_text",
                        category=category,
                        tags = tags
                    ))
                except ValidationError as e:
                    invalid_count += 1
                    logger.warning("Invalid entry at index %s: %s", idx, e)
                    continue

            # No generation task defined for this dataset yet
        
        if invalid_count:
            logger.warning("Skipped %s invalid entries during MedXpertQA load.", invalid_count)
        print(f"Loaded {len(samples)}/{len(raw_data)} MedXpertQA samples.")
        return samples

if __name__ == "__main__":
    loader = MedXpertLoader()
    data = loader.load()
    if data:
        print(f"\nFirst Sample: {data[0]}")
        print(f"\nCategory: {data[0].category}")
        print(f"\nData Type: {type(data[0])}")
