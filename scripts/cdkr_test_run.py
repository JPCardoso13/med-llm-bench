from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from llm_bench.backends import OpenAIBackend
from llm_bench.formatters import GenerativeFormatter, MCQFormatter
from llm_bench.ingestion import YamlLoader
from llm_bench.metrics import calculate_system_metrics
from llm_bench.runner import SequentialRunner
from llm_bench.telemetry import (
    NvidiaSmiTelemetryCollector,
    NullTelemetryCollector,
    RemoteHttpTelemetryCollector,
)


TASK_CONFIG_PATH = Path("configs/tasks/cdkr.yaml")
MODEL_CONFIG_PATH = Path("configs/models/llama3_8b_instruct.yaml")
PER_DATASET_EVAL_LIMIT = 5


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dotenv(path: str | Path = ".env") -> None:
    dotenv_path = Path(path)
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def resolve_runtime_config_path(task_cfg: dict[str, Any]) -> Path:
    return Path(task_cfg["runtime"]["telemetry_config"])


def build_telemetry_collector(runtime_config: dict[str, Any]):
    telemetry_cfg = runtime_config.get("telemetry", {})
    if not telemetry_cfg.get("enabled", False):
        return NullTelemetryCollector()

    collector_name = telemetry_cfg.get("collector", "remote_http")

    if collector_name == "nvidia_smi":
        return NvidiaSmiTelemetryCollector(
            gpu_indices=telemetry_cfg.get("gpu_indices"),
            poll_interval_s=telemetry_cfg.get("poll_interval_s", 0.5),
        )

    if collector_name == "remote_http":
        endpoints = telemetry_cfg.get("endpoints", [])
        if not endpoints:
            raise ValueError("remote_http telemetry requires a non-empty endpoints list")
        return RemoteHttpTelemetryCollector(
            endpoints=endpoints,
            window_path=telemetry_cfg.get("window_path", "/window"),
            timeout_s=telemetry_cfg.get("timeout_s", 1.5),
        )

    raise ValueError(f"Unsupported telemetry collector: {collector_name}")


def build_backend(model_cfg: dict[str, Any]) -> OpenAIBackend:
    model_id = str(model_cfg["model_id"])
    base_url = os.environ["LLM_BASE_URL"]

    api_key = os.getenv("LLM_API_KEY", "EMPTY")
    temperature = float(model_cfg.get("temperature", 0.0))
    max_tokens = int(model_cfg.get("max_tokens", 512))

    return OpenAIBackend(
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def build_formatter(task_cfg: dict[str, Any]):
    prompt_cfg = load_yaml(task_cfg["prompt"]["config"])
    task_type = task_cfg.get("task_type", "mcq")

    if task_type == "mcq":
        return MCQFormatter(
            system_prompt=prompt_cfg["system_prompt"],
            user_turn_template=prompt_cfg["user_turn_template"],
            fewshot_template=prompt_cfg.get("fewshot_template"),
            fewshot_header=prompt_cfg.get("fewshot_header"),
        )

    if task_type == "generative":
        return GenerativeFormatter(
            system_prompt=prompt_cfg["system_prompt"],
            user_turn_template=prompt_cfg["user_turn_template"],
            fewshot_template=prompt_cfg.get("fewshot_template"),
        )

    raise ValueError(f"Unsupported task_type: {task_type}")


def resolve_raw_output_path(task_cfg: dict[str, Any], dataset_name: str) -> Path:
    outputs_cfg = task_cfg.get("outputs", {})
    raw_dir = Path(outputs_cfg.get("raw_dir", "outputs"))
    pattern = outputs_cfg.get("raw_file_pattern", "{dataset}_test.jsonl")
    return raw_dir / pattern.format(dataset=dataset_name)


def resolve_systems_summary_json_path(task_cfg: dict[str, Any]) -> Path:
    outputs_cfg = task_cfg.get("outputs", {})
    task_id = task_cfg.get("task_id", "task")
    template = outputs_cfg.get("systems_summary_json", "outputs/reports/{task_id}_systems_summary.json")
    return Path(template.format(task_id=task_id))


def main() -> None:
    load_dotenv()

    task_cfg = load_yaml(TASK_CONFIG_PATH)
    systems_profile = load_yaml(task_cfg["metrics"]["systems_profile"])

    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    print(f"Using model config: {MODEL_CONFIG_PATH}")

    runtime_cfg_path = resolve_runtime_config_path(task_cfg)
    print(f"Using runtime telemetry config: {runtime_cfg_path}")
    runtime_cfg = load_yaml(runtime_cfg_path)

    backend = build_backend(model_cfg)
    formatter = build_formatter(task_cfg)
    telemetry_collector = build_telemetry_collector(runtime_cfg)

    num_fewshot = int(task_cfg.get("execution", {}).get("num_fewshot", 0))
    flush_every = int(task_cfg.get("execution", {}).get("flush_every", 10))
    task_name = task_cfg.get("task_id", "cdkr")

    all_results = []

    datasets_cfg = task_cfg.get("datasets", [])
    enabled_datasets = [d for d in datasets_cfg if d.get("enabled", True)]

    for ds in enabled_datasets:
        dataset_name = ds["name"]
        dataset_config_path = ds["config"]

        print(f"\n--- Running dataset: {dataset_name} ---")
        loader = YamlLoader(dataset_config_path)
        data = loader.load()

        eval_samples = data.get("eval", [])[:PER_DATASET_EVAL_LIMIT]
        fewshot_samples = data.get("fewshot", [])

        output_path = resolve_raw_output_path(task_cfg, dataset_name)

        runner = SequentialRunner(
            backend=backend,
            formatter=formatter,
            task_name=task_name,
            dataset_name=dataset_name,
            output_path=output_path,
            num_fewshot=num_fewshot,
            fewshot_pool=fewshot_samples,
            flush_every=flush_every,
            telemetry_collector=telemetry_collector,
        )

        results = runner.run(eval_samples)
        all_results.extend(results)

        dataset_summary = calculate_system_metrics(results, systems_profile)
        print(f"Systems summary for {dataset_name}: groups={len(dataset_summary.get('groups', []))}")

    overall_summary = calculate_system_metrics(all_results, systems_profile)
    summary_path = resolve_systems_summary_json_path(task_cfg)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

    print("\n=== CDKR test run complete ===")
    print(f"Total results: {len(all_results)}")
    print(f"Systems summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
