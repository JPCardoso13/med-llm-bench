import logging
import uuid
from typing import List, Optional
from pydantic import ValidationError
from .base_loader import BaseLoader, BenchmarkSample
from llm_bench.schemas import GenerativeSample

logger = logging.getLogger(__name__)

class MedCaseReasoningLoader(BaseLoader):
	"""
	Loader for the MedCaseReasoning dataset.

	This loader produces GenerativeSample objects with:
	- question <- case_prompt
	- answer <- final_diagnosis
	- ref_reasoning <- diagnostic_reasoning
	- context <- text (full source case text)

	Attributes:
		path_or_name (str): Local file path or dataset identifier.
							Defaults to 'zou-lab/MedCaseReasoning'.
		task_type (str): Task format to load. Must be "generation" for this loader.
		split (Optional[str]): Dataset split name if applicable. Defaults to "train".
	"""

	def __init__(
		self,
		path_or_name: str = "zou-lab/MedCaseReasoning",
		task_type: str = "generation",
		split: Optional[str] = "train",
	):
		super().__init__(path_or_name, task_type, split)

		if self.task_type != "generation":
			raise ValueError(
				"MedCaseReasoningLoader only supports task_type='generation'."
			)

	def load(self) -> List[BenchmarkSample]:
		print(
			f"Loading MedCaseReasoning from {self.path_or_name} "
			f"(Task: {self.task_type}, Split: {self.split})..."
		)

		if self._is_local_source():
			raw_data = self._read_jsonl()
		else:
			from datasets import load_dataset

			raw_data = load_dataset(self.path_or_name, split=self.split)

		samples: List[BenchmarkSample] = []
		print(f"Processing {len(raw_data)} entries...")

		invalid_count = 0
		for idx, entry in enumerate(raw_data):
			pmcid = entry.get("pmcid")
			unique_id = (
				f"medcasereasoning_{pmcid}" if pmcid else f"medcasereasoning_{str(uuid.uuid4())[:8]}"
			)

			tags = {
				"pmcid": str(pmcid) if pmcid is not None else "",
				"title": str(entry.get("title") or ""),
				"journal": str(entry.get("journal") or ""),
				"publication_date": str(entry.get("publication_date") or ""),
				"article_link": str(entry.get("article_link") or ""),
				"split": str(self.split or ""),
			}

			try:
				samples.append(
					GenerativeSample(
						id=unique_id,
						question=entry.get("case_prompt"),
						answer=entry.get("final_diagnosis"),
						ref_reasoning=entry.get("diagnostic_reasoning"),
						source="medcasereasoning",
						category=entry.get("journal"),
						tags=tags,
						context=entry.get("text"),
					)
				)
			except ValidationError as e:
				invalid_count += 1
				logger.warning("Invalid entry at index %s: %s", idx, e)
				continue

		if invalid_count:
			logger.warning(
				"Skipped %s invalid entries during MedCaseReasoning load.", invalid_count
			)

		print(f"Loaded {len(samples)}/{len(raw_data)} MedCaseReasoning samples.")
		return samples


if __name__ == "__main__":
	loader = MedCaseReasoningLoader()
	data = loader.load()
	if data:
		print(f"\nFirst Sample: {data[0]}")
		print(f"\nCategory: {data[0].category}")
		print(f"\nData Type: {type(data[0])}")
