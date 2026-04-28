from llm_bench.ingestion import YamlLoader
from llm_bench.backends import OpenAIBackend
from llm_bench.formatters import MCQFormatter
from llm_bench.runner import SequentialRunner
from llm_bench.telemetry import (
    NvidiaSmiTelemetryCollector,
    NullTelemetryCollector,
    RemoteHttpTelemetryCollector,
)
import os
import yaml
from pathlib import Path


def load_prompt_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_runtime_config(path: str) -> dict:
    runtime_path = Path(path)
    if not runtime_path.exists():
        return {}

    with open(runtime_path, "r") as f:
        return yaml.safe_load(f) or {}


def build_telemetry_collector(config: dict):
    telemetry_cfg = config.get("telemetry", {})
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
            raise ValueError("remote_http telemetry collector requires a non-empty endpoints list.")

        return RemoteHttpTelemetryCollector(
            endpoints=endpoints,
            window_path=telemetry_cfg.get("window_path", "/window"),
            timeout_s=telemetry_cfg.get("timeout_s", 1.5),
        )

    raise ValueError(f"Unsupported telemetry collector: {collector_name}")

runtime_cfg_path = os.getenv("LLM_BENCH_RUNTIME_CONFIG")
if runtime_cfg_path is None:
    auto_cfg_path = Path("configs/runtime/telemetry.auto.yaml")
    runtime_cfg_path = str(auto_cfg_path if auto_cfg_path.exists() else Path("configs/runtime/telemetry.yaml"))

runtime_cfg = load_runtime_config(runtime_cfg_path)
telemetry_collector = build_telemetry_collector(runtime_cfg)

backend = OpenAIBackend(
    model_id="Qwen/Qwen3-32B-AWQ",
    base_url="http://192.168.112.13:8000/v1",
    api_key="EMPTY",
    temperature=0.0,
    max_tokens=512,
)

prompt_cfg = load_prompt_config("configs/prompts/cdkr.yaml")
formatter = MCQFormatter(
    system_prompt=prompt_cfg["system_prompt"],
    user_turn_template=prompt_cfg["user_turn_template"],
    fewshot_template=prompt_cfg.get("fewshot_template"),
    fewshot_header=prompt_cfg.get("fewshot_header"),
)

datasets = [
    {
        "config": "configs/datasets/medqa.yaml",
        "name": "medqa",
        "output": "outputs/medqa_test.json",
    },
    {
        "config": "configs/datasets/medxpertqa.yaml",
        "name": "medxpertqa",
        "output": "outputs/medxpertqa_test.json",
    },
]

for ds in datasets:
    print(f"\n--- Running dataset: {ds['name']} ---")

    loader = YamlLoader(ds["config"])
    data = loader.load()
    eval_samples = data["eval"]
    fewshot_samples = data["fewshot"]

    runner = SequentialRunner(
        backend=backend,
        formatter=formatter,
        task_name="closed_domain_knowledge_retrieval",
        dataset_name=ds["name"],
        output_path=ds["output"],
        num_fewshot=3,
        fewshot_pool=fewshot_samples,
        flush_every=10,
        telemetry_collector=telemetry_collector,
    )

    results = runner.run(eval_samples[:10])
    print(f"Completed {ds['name']}: {len(results)} results")
    print(f"Sample response: {results[0].response[:100]}")