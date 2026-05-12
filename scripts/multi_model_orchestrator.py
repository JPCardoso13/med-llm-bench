from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from vllm_manager import start_vllm, stop_vllm
from llm_bench.backends import OpenAIBackend
from llm_bench.formatters import GenerativeFormatter, MCQFormatter
from llm_bench.ingestion import YamlLoader
from llm_bench.metrics import calculate_cognitive_metrics, calculate_system_metrics
from llm_bench.runner import SequentialRunner
from llm_bench.schemas import BenchmarkResult
from llm_bench.telemetry import (
    NvidiaSmiTelemetryCollector,
    NullTelemetryCollector,
    RemoteHttpTelemetryCollector,
)
from llm_bench.utils.io import save_results_json


TASKS_DIR = Path("configs/tasks")
MODELS_DIR = Path("configs/models")
PER_DATASET_EVAL_LIMIT = 5
RAW_RESULTS_DIR = Path("outputs/raw")
REPORTS_DIR = Path("outputs/reports")
TMP_RESULTS_DIR = Path("outputs/tmp")
SERVE_LOG_DIR = Path("logs/vllm")


def maybe_evict_hf_cache_between_models() -> None:
    if os.getenv("HF_EVICT_BETWEEN_MODELS", "0") != "1":
        return

    hf_home_raw = os.getenv("HF_HOME")
    if not hf_home_raw:
        return

    hf_home = Path(hf_home_raw)
    if not hf_home.exists() or not hf_home.is_dir():
        return

    removed_entries = 0
    for child in hf_home.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
            removed_entries += 1
        except Exception as exc:
            print(f"  Warning: failed to remove cache entry {child}: {exc}")

    hf_home.mkdir(parents=True, exist_ok=True)
    print(f"  Cleared Hugging Face cache entries: {removed_entries}")


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
            window_padding_s=telemetry_cfg.get("window_padding_s", 0.5),
        )

    raise ValueError(f"Unsupported telemetry collector: {collector_name}")


def build_backend(model_cfg: dict[str, Any], base_url: str, max_tokens: int = 1024) -> OpenAIBackend:
    model_id = str(model_cfg["model_id"])

    api_key = os.getenv("LLM_API_KEY", "EMPTY")
    temperature = float(model_cfg.get("temperature", 0.0))

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


def discover_task_configs() -> list[Path]:
    """Discover all task configs from configs/tasks/*.yaml, sorted by name."""
    configs = list(TASKS_DIR.glob("*.yaml"))
    configs = [c for c in configs if c.stem != "template"]
    return sorted(configs)


def discover_model_configs() -> list[Path]:
    return sorted(MODELS_DIR.glob("*.yaml"))


def run_benchmark_for_model(
    model_cfg: dict[str, Any],
    model_name: str,
    task_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    base_url: str,
    task_id: str,
) -> Path:
    max_tokens = int(task_cfg.get("execution", {}).get("max_tokens", 1024))
    backend = build_backend(model_cfg, base_url, max_tokens=max_tokens)
    formatter = build_formatter(task_cfg)
    telemetry_collector = build_telemetry_collector(runtime_cfg)

    num_fewshot = int(task_cfg.get("execution", {}).get("num_fewshot", 0))
    flush_every = int(task_cfg.get("execution", {}).get("flush_every", 10))
    task_name = task_cfg.get("task_id", "task")

    all_results = []
    datasets_cfg = task_cfg.get("datasets", [])
    enabled_datasets = [d for d in datasets_cfg if d.get("enabled", True)]

    for ds in enabled_datasets:
        dataset_name = ds["name"]
        dataset_config_path = ds["config"]

        loader = YamlLoader(dataset_config_path)
        data = loader.load()

        eval_limit = ds.get("eval_limit")
        if eval_limit is None:
            eval_limit = PER_DATASET_EVAL_LIMIT

        eval_samples = data.get("eval", [])[: int(eval_limit)]
        fewshot_samples = data.get("fewshot", [])

        tmp_output_path = TMP_RESULTS_DIR / task_id / model_name / f"{dataset_name}.json"
        tmp_output_path.parent.mkdir(parents=True, exist_ok=True)

        runner = SequentialRunner(
            backend=backend,
            formatter=formatter,
            task_name=task_name,
            dataset_name=dataset_name,
            output_path=tmp_output_path,
            num_fewshot=num_fewshot,
            fewshot_pool=fewshot_samples,
            flush_every=flush_every,
            telemetry_collector=telemetry_collector,
        )

        results = runner.run(eval_samples)
        all_results.extend(results)
        print(f"  Dataset {dataset_name}: {len(results)}/{len(eval_samples)} results")

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_RESULTS_DIR / task_id / f"{model_name}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    save_results_json(all_results, raw_path)
    print(f"  Raw results: {raw_path}")
    return raw_path


def compute_and_save_metrics(model_name: str, raw_path: Path, task_id: str, systems_profile: dict[str, Any], cognitive_profile: dict[str, Any]) -> tuple[Path, Path | None]:
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    all_results = [BenchmarkResult(**r) for r in raw_data]

    overall_summary = calculate_system_metrics(all_results, systems_profile)
    task_report_dir = REPORTS_DIR / task_id / "model_summaries"
    task_report_dir.mkdir(parents=True, exist_ok=True)
    
    systems_summary_path = task_report_dir / f"{model_name}_systems_summary.json"
    systems_summary_path.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

    cognitive_summary_path = None
    if cognitive_profile.get("enabled", True):
        cognitive_overall_summary = calculate_cognitive_metrics(all_results, cognitive_profile)
        cognitive_summary_path = task_report_dir / f"{model_name}_cognitive_summary.json"
        cognitive_summary_path.write_text(json.dumps(cognitive_overall_summary, indent=2), encoding="utf-8")

    return systems_summary_path, cognitive_summary_path


def run_single_task(task_cfg_path: Path, model_configs: list[Path], serve_port: int, startup_timeout_s: int) -> dict[str, Any]:
    """Run all models for a single task."""
    task_cfg = load_yaml(task_cfg_path)
    task_id = task_cfg.get("task_id")
    
    if not task_id:
        raise ValueError(f"Task config {task_cfg_path} missing 'task_id' field")

    print(f"\n{'='*80}")
    print(f"TASK: {task_id}")
    print(f"{'='*80}")

    systems_profile = load_yaml(task_cfg["metrics"]["systems_profile"])
    cognitive_profile_path = task_cfg.get("metrics", {}).get("cognitive_profile")
    cognitive_profile = load_yaml(cognitive_profile_path) if cognitive_profile_path else {"enabled": False}

    runtime_cfg_path = resolve_runtime_config_path(task_cfg)
    runtime_cfg = load_yaml(runtime_cfg_path)

    summary = {"task_id": task_id, "models": []}

    for model_config_path in model_configs:
        model_name = model_config_path.stem
        model_cfg = load_yaml(model_config_path)

        print(f"\n== Model: {model_name} ==")
        handle = None
        try:
            model_startup_timeout_s = int(model_cfg.get("startup_timeout_s", startup_timeout_s))
            handle = start_vllm(
                model_cfg=model_cfg,
                port=serve_port,
                logs_dir=SERVE_LOG_DIR,
                timeout_s=model_startup_timeout_s,
            )
            print(f"  Serving at {handle.base_url} (mode={handle.mode})")

            os.environ["LLM_BASE_URL"] = handle.base_url
            raw_path = run_benchmark_for_model(model_cfg, model_name, task_cfg, runtime_cfg, handle.base_url, task_id)

            systems_path, cognitive_path = compute_and_save_metrics(model_name, raw_path, task_id, systems_profile, cognitive_profile)
            print(f"  Systems summary: {systems_path}")
            if cognitive_path:
                print(f"  Cognitive summary: {cognitive_path}")

            summary["models"].append({
                "model_name": model_name,
                "raw_results": str(raw_path),
                "systems_summary": str(systems_path),
                "cognitive_summary": str(cognitive_path) if cognitive_path else None,
                "serve_mode": handle.mode,
            })
        except Exception as exc:
            print(f"ERROR processing {model_name}: {exc}")
            summary["models"].append({"model_name": model_name, "error": str(exc)})
        finally:
            stop_vllm(handle)
            time.sleep(8)
            maybe_evict_hf_cache_between_models()

    # Write task-specific summary
    task_report_dir = REPORTS_DIR / task_id
    task_report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = task_report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nTask report written to: {summary_path}")

    return summary


def main() -> None:
    load_dotenv()

    task_configs = discover_task_configs()
    if not task_configs:
        print("No task configs found in configs/tasks/ (excluding template.yaml)")
        return

    model_configs = discover_model_configs()
    if not model_configs:
        print("No model configs found in configs/models/")
        return

    serve_port = int(os.getenv("SERVE_PORT", "8000"))
    startup_timeout_s = int(os.getenv("VLLM_STARTUP_TIMEOUT_S", "1800"))

    all_summaries = {"tasks": []}

    for task_cfg_path in task_configs:
        try:
            task_summary = run_single_task(task_cfg_path, model_configs, serve_port, startup_timeout_s)
            all_summaries["tasks"].append(task_summary)
        except Exception as exc:
            print(f"ERROR processing task {task_cfg_path.stem}: {exc}")
            all_summaries["tasks"].append({"task_id": task_cfg_path.stem, "error": str(exc)})

    # Write overall multi-task summary
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    overall_summary_path = REPORTS_DIR / "multi_task_summary.json"
    overall_summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"\n{'='*80}")
    print(f"Overall multi-task summary written to: {overall_summary_path}")
    print(f"{'='*80}")

    failed_tasks = [t for t in all_summaries["tasks"] if "error" in t]
    print(f"\nRun complete")
    print(f"Successful tasks: {len(all_summaries['tasks']) - len(failed_tasks)} | Failed tasks: {len(failed_tasks)}")

    if failed_tasks:
        sys.exit(1)


if __name__ == "__main__":
    main()
