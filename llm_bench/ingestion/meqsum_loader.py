import logging
from typing import ClassVar, List, Optional, Set
from pydantic import ValidationError
from datasets import load_dataset

from .base_loader import BaseLoader, BenchmarkSample
from llm_bench.schemas import GenerativeSample

logger = logging.getLogger(__name__)


MEQSUM_PARQUET_URLS = {
    "meqsum_bigbio_t2t": "https://huggingface.co/datasets/bigbio/meqsum/resolve/refs%2Fconvert%2Fparquet/meqsum_bigbio_t2t/train/0000.parquet",
    "meqsum_source": "https://huggingface.co/datasets/bigbio/meqsum/resolve/refs%2Fconvert%2Fparquet/meqsum_source/train/0000.parquet",
}


class MeQSumLoader(BaseLoader):
    """
    Loader for the MeQSum dataset.

    Attributes:
        path_or_name (str): Local file path or dataset identifier.
                            Defaults to 'bigbio/meqsum'.
        task_type (str): Task format to load. Must be "generation" for this loader.
        subset (Optional[str]): Dataset subset name if applicable. Defaults to "meqsum_bigbio_t2t".
        split (Optional[str]): Dataset split name if applicable. Defaults to "train".
    """

    ALLOWED_TASK_TYPES: ClassVar[Set[str]] = {"generation"}

    def __init__(
        self,
        path_or_name: str = "bigbio/meqsum",
        task_type: str = "generation",
        subset: Optional[str] = "meqsum_bigbio_t2t",
        split: Optional[str] = "train",
    ):
        super().__init__(
            path_or_name=path_or_name,
            task_type=task_type,
            subset=subset,
            split=split,
        )

    def load(self) -> List[BenchmarkSample]:
        logger.info(
            f"Loading MeQSum from {self.path_or_name}..."
            f"(Task: {self.task_type}, Subset: {self.subset}, Split: {self.split})"
        )

        source = self.path_or_name.strip()
        split = self.split or "train"

        if self._is_local_source():
            raw_data = self._read_jsonl()
        else:
            try:
                raw_data = load_dataset(source, name=self.subset, split=split)
            except RuntimeError as e:
                if source == "bigbio/meqsum" and "Dataset scripts are no longer supported" in str(e):
                    parquet_url = MEQSUM_PARQUET_URLS.get(self.subset or "")
                    if not parquet_url:
                        raise RuntimeError(
                            "Unsupported MeQSum subset for parquet fallback: "
                            f"{self.subset}. Supported subsets: {list(MEQSUM_PARQUET_URLS.keys())}"
                        ) from e

                    logger.warning(
                        "Falling back to MeQSum parquet export for subset '%s'.",
                        self.subset,
                    )
                    raw_data = load_dataset("parquet", data_files={split: parquet_url}, split=split)
                else:
                    raise  # In case it errors for another reason

        samples: List[BenchmarkSample] = []
        logger.info("Processing %s entries...", len(raw_data))

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

        logger.info("Loaded %s/%s MeQSum samples.", len(samples), len(raw_data))
        return samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = MeQSumLoader()
    loaded_samples = loader.load()

    first_entry = loaded_samples[0]
    print("First loaded sample:\n", first_entry)
    print("Type of first loaded sample:\n", type(first_entry))
