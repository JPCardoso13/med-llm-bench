import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_bench.backends.base_backend import BaseBackend
from llm_bench.formatters.base_formatter import BaseFormatter
from llm_bench.schemas import BenchmarkResult, GenerativeSample, MCQSample
from llm_bench.telemetry.base_collector import TelemetryCollector
from llm_bench.telemetry.null_collector import NullTelemetryCollector
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
        telemetry_collector: Optional[TelemetryCollector] = None,
    ):
        self._backend = backend
        self._formatter = formatter
        self._task_name = task_name
        self._dataset_name = dataset_name
        self._output_path = Path(output_path)
        self._num_fewshot = num_fewshot
        self._fewshot_pool = fewshot_pool or []
        self._flush_every = flush_every
        self._telemetry_collector = telemetry_collector or NullTelemetryCollector()

    def run(self, samples: List[Sample]) -> List[BenchmarkResult]:
        if self._output_path.exists():
            self._output_path.unlink()

        run_context = {
            "task_name": self._task_name,
            "dataset_name": self._dataset_name,
            "backend": self._backend.backend_name,
            "model_id": self._backend.model_id,
            "sample_count": len(samples),
        }
        self._telemetry_collector.start_run(run_context)

        results = []
        pending = []
        telemetry_warning_count = 0
        suppressed_warning_notice_printed = False

        try:
            for i, sample in enumerate(samples):
                fewshot_examples = self._sample_fewshot()
                formatted_user_turn = self._formatter.format(sample, fewshot_examples)

                messages = [
                    {"role": "system", "content": self._formatter.system_prompt},
                    {"role": "user", "content": formatted_user_turn},
                ]

                ref_fields = self._extract_ref_fields(sample)

                request_context: Dict[str, Any] = {
                    "sample_id": sample.id,
                    "sample_index": i,
                    "dataset_name": self._dataset_name,
                    "task_name": self._task_name,
                    "sample_type": self._resolve_sample_type(sample),
                    "request_started_at_epoch_s": time.time(),
                    "request_started_at_perf_counter": time.perf_counter(),
                }
                self._telemetry_collector.before_request(request_context)

                result = self._backend.generate(
                    messages=messages,
                    sample_id=sample.id,
                    dataset=self._dataset_name,
                    task_name=self._task_name,
                    sample_type=request_context["sample_type"],
                    ref_fields=ref_fields,
                )

                request_context["request_finished_at_epoch_s"] = time.time()
                request_context["request_finished_at_perf_counter"] = time.perf_counter()
                request_context["request_duration_ms"] = (
                    request_context["request_finished_at_perf_counter"]
                    - request_context["request_started_at_perf_counter"]
                ) * 1000

                system_metrics = self._telemetry_collector.after_request(request_context)
                if system_metrics:
                    result.backend_metrics.setdefault("system", {}).update(system_metrics)

                capabilities = result.backend_metrics.setdefault("capabilities", {})
                capabilities.setdefault("request_timing", True)
                capabilities.setdefault("token_usage", "usage" in result.backend_metrics)
                capabilities["system_telemetry"] = bool(system_metrics)
                capabilities["system_telemetry_ok"] = not self._has_telemetry_issue(system_metrics)

                if self._has_telemetry_issue(system_metrics):
                    telemetry_warning_count += 1
                    if telemetry_warning_count <= 3:
                        print(
                            "WARNING telemetry issue "
                            f"sample_id={sample.id}: {self._summarize_telemetry_issue(system_metrics)}"
                        )
                    elif not suppressed_warning_notice_printed:
                        print(
                            "WARNING additional telemetry issues detected; "
                            "further per-sample warnings are suppressed."
                        )
                        suppressed_warning_notice_printed = True

                results.append(result)
                pending.append(result)

                if len(pending) >= self._flush_every:
                    save_results_jsonl(pending, self._output_path)
                    pending = []
                    print(f"Flushed results at sample {i + 1}/{len(samples)}")
        finally:
            run_summary = self._telemetry_collector.stop_run(run_context)
            if run_summary:
                print(f"Telemetry run summary: {run_summary}")

        if pending:
            save_results_jsonl(pending, self._output_path)

        print(f"Run complete. {len(results)} results saved to {self._output_path}")
        return results

    def _sample_fewshot(self) -> List[Sample]:
        if self._num_fewshot == 0 or not self._fewshot_pool:
            return []
        return random.sample(
            self._fewshot_pool,
            min(self._num_fewshot, len(self._fewshot_pool)),
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

    def _has_telemetry_issue(self, system_metrics: Dict[str, Any]) -> bool:
        if not system_metrics:
            return False

        if system_metrics.get("node_errors"):
            return True

        if "sample_count" in system_metrics:
            return int(system_metrics.get("sample_count", 0)) == 0

        if "sample_count_total" in system_metrics:
            return int(system_metrics.get("sample_count_total", 0)) == 0

        return False

    def _summarize_telemetry_issue(self, system_metrics: Dict[str, Any]) -> str:
        node_errors = system_metrics.get("node_errors")
        if node_errors:
            first_error = node_errors[0]
            endpoint = first_error.get("endpoint", "unknown")
            message = first_error.get("error", "unknown error")
            return f"node_error endpoint={endpoint} error={message}"

        if "sample_count_total" in system_metrics:
            return "telemetry returned zero samples across nodes"

        if "sample_count" in system_metrics:
            return "telemetry returned zero samples"

        return "telemetry returned an unknown issue"