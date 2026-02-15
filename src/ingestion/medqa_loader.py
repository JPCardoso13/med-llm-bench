import uuid
from typing import List
from src.ingestion.base_loader import BaseLoader, BenchmarkSample
from src.schemas.mcqsample import MCQSample
from src.schemas.generativesample import GenerativeSample
from pydantic import ValidationError

class MedQALoader(BaseLoader):
    """
    Loader for MedQA.
    """

    def load(self) -> List[BenchmarkSample]:
        raw_data = self._read_jsonl()
        samples = []
        
        print(f"Loading MedQA from {self.file_path} (Task: {self.task_type})...")

        for entry in raw_data:
            # ID is absent in the raw dataset, so we generate a unique one for each sample.
            unique_id = f"medqa_{str(uuid.uuid4())[:8]}"
            
            if self.task_type == "mcq":
                try:
                    samples.append(MCQSample(
                        id=unique_id,
                        question=entry.get('question'),
                        options=entry.get('options'),
                        answer_idx=entry.get('answer_idx'),
                        source="medqa",
                        category=entry.get('meta_info')
                    ))
                except ValidationError:
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
                except ValidationError:
                    continue

        print(f"Loaded {len(samples)}/{len(raw_data)} MedQA samples.")

        return samples

if __name__ == "__main__":
    loader = MedQALoader(file_path="data/processed/medqa/fewshots.jsonl", task_type="mcq")
    data = loader.load()
    if data:
        print(f"First Sample: {data[0]}")
        print(f"Data Type: {type(data[0])}")
