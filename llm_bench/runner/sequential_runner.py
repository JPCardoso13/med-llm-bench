import random
from pathlib import Path
from typing import List, Optional
from llm_bench.backends.base_backend import BaseBackend
from llm_bench.prompt.base_formatter import BaseFormatter
from llm_bench.schemas.mcq_sample import MCQSample
from llm_bench.schemas.generative_sample import GenerativeSample
from llm_bench.schemas.benchmark_result import BenchmarkResult
from llm_bench.utils.io import save_results_jsonl

Sample = MCQSample | GenerativeSample


class SequentialRunner:

    def __init__(
        self,
        backend: BaseBackend,
        formatter: BaseFormatter,
        task_name: str,
        dataset_name: str,
        output_path: str | Path,
        num_fewshot: int = 0,
        fewshot_pool: Optional[List[Sample]] = None,
        flush_every: int = 10,
    ):
        self._backend = backend
        self._formatter = formatter
        self._task_name = task_name
        self._dataset_name = dataset_name
        self._output_path = Path(output_path)
        self._num_fewshot = num_fewshot
        self._fewshot_pool = fewshot_pool or []
        self._flush_every = flush_every

    def run(self, samples: List[Sample]) -> List[BenchmarkResult]:
        results = []
        pending = []

        for i, sample in enumerate(samples):
            fewshot_examples = self._sample_fewshot()
            formatted_user_turn = self._formatter.format(sample, fewshot_examples)

            messages = [
                {"role": "system", "content": self._formatter.system_prompt},
                {"role": "user", "content": formatted_user_turn},
            ]

            ref_fields = self._extract_ref_fields(sample)

            result = self._backend.generate(
                messages=messages,
                sample_id=sample.id,
                dataset=self._dataset_name,
                task_name=self._task_name,
                sample_type=self._resolve_sample_type(sample),
                ref_fields=ref_fields,
            )

            results.append(result)
            pending.append(result)

            if len(pending) >= self._flush_every:
                save_results_jsonl(pending, self._output_path)
                pending = []
                print(f"Flushed results at sample {i + 1}/{len(samples)}")

        if pending:
            save_results_jsonl(pending, self._output_path)

        print(f"Run complete. {len(results)} results saved to {self._output_path}")
        return results

    def _sample_fewshot(self) -> List[Sample]:
        if self._num_fewshot == 0 or not self._fewshot_pool:
            return []
        return random.sample(
            self._fewshot_pool,
            min(self._num_fewshot, len(self._fewshot_pool))
        )

    def _extract_ref_fields(self, sample: Sample) -> dict:
        if isinstance(sample, MCQSample):
            return {"answer_idx": sample.answer_idx}
        if isinstance(sample, GenerativeSample):
            fields = {"answer": sample.answer}
            if sample.ref_reasoning is not None:
                fields["ref_reasoning"] = sample.ref_reasoning
            return fields
        return {}

    def _resolve_sample_type(self, sample: Sample) -> str:
        if isinstance(sample, MCQSample):
            return "mcq"
        if isinstance(sample, GenerativeSample):
            return "generative"
        return "unknown"