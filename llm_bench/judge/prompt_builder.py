from __future__ import annotations

from typing import Any, Mapping

from jinja2 import BaseLoader, Environment

from llm_bench.schemas.benchmark_result import BenchmarkResult

_env = Environment(loader=BaseLoader())


def build_judge_messages(result: BenchmarkResult, prompt_cfg: Mapping[str, Any]) -> list[dict[str, str]]:
    template_vars = {
        "case_context": result.prompt,
        "model_response": result.response,
        "reference_answer": result.ref_fields.get("answer", ""),
        "reference_reasoning": result.ref_fields.get("ref_reasoning", ""),
    }

    user_turn = _env.from_string(prompt_cfg["user_turn_template"]).render(**template_vars)

    return [
        {"role": "system", "content": prompt_cfg["system_prompt"]},
        {"role": "user", "content": user_turn},
    ]
