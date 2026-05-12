import time
import re
from datetime import datetime
from typing import Any, List, Dict, Optional

from openai import OpenAI, BadRequestError
from llm_bench.schemas import BenchmarkResult
from .base_backend import BaseBackend


class OpenAIBackend(BaseBackend):

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        self._model_id = model_id
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._extra_params = extra_params or {}

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    @property
    def backend_name(self) -> str:
        return "openai_compatible"

    @property
    def model_id(self) -> str:
        return self._model_id

    def _max_tokens_from_error(self, error_message: str) -> Optional[int]:
        # Example message fragment: "(1024 > 4096 - 3510)"
        match = re.search(r"\((\d+)\s*>\s*(\d+)\s*-\s*(\d+)\)", error_message)
        if not match:
            return None

        context_window = int(match.group(2))
        prompt_tokens = int(match.group(3))
        available = context_window - prompt_tokens
        # Keep a small margin for safety and tokenizer variation.
        adjusted = max(1, available - 16)
        return adjusted

    def _create_stream(self, messages: List[Dict[str, str]], max_tokens: int):
        return self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            temperature=self._temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **self._extra_params,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        sample_id: str,
        dataset: str,
        task_name: str,
        sample_type: str,
        ref_fields: dict,
        grouping: dict,
    ) -> BenchmarkResult:

        user_prompt = "\n".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        ).strip()

        ttft_ms = None
        inter_token_latencies_ms = []
        chunks = []
        last_token_time = None
        backend_metrics = {}

        request_start = time.perf_counter()

        chosen_max_tokens = self._max_tokens
        try:
            stream = self._create_stream(messages, chosen_max_tokens)
        except BadRequestError as exc:
            error_text = str(exc)
            if "max_tokens" not in error_text and "max_completion_tokens" not in error_text:
                raise

            adjusted = self._max_tokens_from_error(error_text)
            if adjusted is None or adjusted >= chosen_max_tokens:
                raise

            chosen_max_tokens = adjusted
            stream = self._create_stream(messages, chosen_max_tokens)

        for chunk in stream:
            now = time.perf_counter()

            if chunk.usage is not None:
                backend_metrics["usage"] = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
                continue

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content
            if delta is None:
                continue

            if ttft_ms is None:
                ttft_ms = (now - request_start) * 1000

            if last_token_time is not None:
                inter_token_latencies_ms.append((now - last_token_time) * 1000)

            last_token_time = now
            chunks.append(delta)

        total_latency_ms = (time.perf_counter() - request_start) * 1000
        response = "".join(chunks)
        usage = backend_metrics.get("usage")
        if not usage:
            raise RuntimeError("Backend did not return token usage; this benchmark expects usage to be present.")

        input_tokens = int(usage["prompt_tokens"])
        output_tokens = int(usage["completion_tokens"])

        return BenchmarkResult(
            sample_id=sample_id,
            dataset=dataset,
            task_name=task_name,
            sample_type=sample_type,
            prompt=user_prompt,
            response=response,
            total_latency_ms=total_latency_ms,
            ttft_ms=ttft_ms if ttft_ms is not None else total_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            inter_token_latencies_ms=inter_token_latencies_ms,
            backend_metrics=backend_metrics,
            ref_fields=ref_fields,
            grouping=grouping,
            model_id=self._model_id,
            backend=self.backend_name,
            timestamp=datetime.utcnow(),
        )
