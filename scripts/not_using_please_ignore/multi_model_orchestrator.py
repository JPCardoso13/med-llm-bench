from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from distributed_fallback import describe_fit, start_vllm_server_for_model, stop_vllm_server
from llm_bench.backends import OpenAIBackend
from llm_bench.formatters import GenerativeFormatter, MCQFormatter
from llm_bench.ingestion import YamlLoader
from llm_bench.metrics import calculate_cognitive_metrics, calculate_system_metrics
from llm_bench.runner import SequentialRunner
from llm_bench.telemetry import (
    NvidiaSmiTelemetryCollector,
    NullTelemetryCollector,
    RemoteHttpTelemetryCollector,
)
from llm_bench.utils.io import save_results_json


TASK_CONFIG_PATH = Path("configs/tasks/cdkr.yaml")
MODELS_DIR = Path("configs/models")
PER_DATASET_EVAL_LIMIT = 5
RAW_RESULTS_DIR = Path("outputs/raw")
REPORTS_DIR = Path("outputs/reports")
TMP_RESULTS_DIR = Path("outputs/tmp")
SERVE_LOG_DIR = Path("logs/vllm_serve")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json_array(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def build_backend(model_cfg: dict[str, Any], base_url: str) -> OpenAIBackend:
    model_id = str(model_cfg["model_id"])

    api_key = os.getenv("LLM_API_KEY", "EMPTY")
    temperature = float(model_cfg.get("temperature", 0.0))
    default_max_tokens = int(os.getenv("LLM_MAX_TOKENS_DEFAULT", "1024"))
    max_tokens = int(model_cfg.get("max_tokens", default_max_tokens))

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


def discover_model_configs() -> list[Path]:
    return sorted(MODELS_DIR.glob("*.yaml"))


def run_benchmark_for_model(
    model_cfg: dict[str, Any],
    model_name: str,
    task_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    base_url: str,
) -> Path:
    backend = build_backend(model_cfg, base_url)
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

        tmp_output_path = TMP_RESULTS_DIR / model_name / f"{dataset_name}.json"
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
        print(f"  Dataset {dataset_name}: {len(results)} results")

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_RESULTS_DIR / f"{model_name}.json"
    save_results_json(all_results, raw_path)
    print(f"  Raw results: {raw_path}")
    return raw_path


def resolve_systems_summary_json_path(task_cfg: dict[str, Any], model_name: str) -> Path:
    task_id = task_cfg.get("task_id", "task")
    return REPORTS_DIR / f"{task_id}_{model_name}_systems_summary.json"


def resolve_cognitive_summary_json_path(task_cfg: dict[str, Any], model_name: str) -> Path:
    task_id = task_cfg.get("task_id", "task")
    return REPORTS_DIR / f"{task_id}_{model_name}_cognitive_summary.json"


def compute_metrics_for_model(
    model_name: str,
    raw_result_path: Path,
    task_cfg: dict[str, Any],
    systems_profile: dict[str, Any],
    cognitive_profile: dict[str, Any],
) -> tuple[Path, Path | None]:
    all_results = load_json_array(raw_result_path)

    overall_summary = calculate_system_metrics(all_results, systems_profile)
    systems_summary_path = resolve_systems_summary_json_path(task_cfg, model_name)
    systems_summary_path.parent.mkdir(parents=True, exist_ok=True)
    systems_summary_path.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

    cognitive_summary_path = None
    if cognitive_profile.get("enabled", True):
        cognitive_overall_summary = calculate_cognitive_metrics(all_results, cognitive_profile)
        cognitive_summary_path = resolve_cognitive_summary_json_path(task_cfg, model_name)
        cognitive_summary_path.parent.mkdir(parents=True, exist_ok=True)
        cognitive_summary_path.write_text(json.dumps(cognitive_overall_summary, indent=2), encoding="utf-8")

    return systems_summary_path, cognitive_summary_path


def main() -> None:
    load_dotenv()

    task_cfg = load_yaml(TASK_CONFIG_PATH)
    systems_profile = load_yaml(task_cfg["metrics"]["systems_profile"])
    cognitive_profile_path = task_cfg.get("metrics", {}).get("cognitive_profile")
    cognitive_profile = load_yaml(cognitive_profile_path) if cognitive_profile_path else {"enabled": False}

    runtime_cfg_path = resolve_runtime_config_path(task_cfg)
    runtime_cfg = load_yaml(runtime_cfg_path)

    model_configs = discover_model_configs()
    if not model_configs:
        print("No model configs found in configs/models/")
        return

    serve_port = int(os.getenv("SERVE_PORT", "8000"))
    startup_timeout_s = int(os.getenv("VLLM_STARTUP_TIMEOUT_S", "360"))

    summary = {
        "strategy": "deferred_only",
        "models": [],
    }

    raw_result_map: dict[str, Path] = {}

    print("=" * 80)
    print("PHASE 1: Benchmark all models and capture raw measurements")
    print("=" * 80)

    for model_config_path in model_configs:
        model_name = model_config_path.stem
        model_cfg = load_yaml(model_config_path)

        print(f"\\n[MODEL] {model_name}")
        fit = describe_fit(model_cfg)
        print(
            "  Fit check: "
            f"requested_tp={fit['requested_tp']}, "
            f"local_gpus={fit['local_gpus']}, "
            f"cluster_gpus={fit['cluster_gpus']}, "
            f"single_possible={fit['single_possible']}, "
            f"distributed_possible={fit['distributed_possible']}"
        )

        handle = None
        try:
            handle = start_vllm_server_for_model(
                model_cfg=model_cfg,
                serve_port=serve_port,
                logs_dir=SERVE_LOG_DIR,
                startup_timeout_s=startup_timeout_s,
            )
            print(f"  Serving mode: {handle.mode}")
            print(f"  Endpoint: {handle.base_url}")

            os.environ["LLM_BASE_URL"] = handle.base_url
            raw_path = run_benchmark_for_model(
                model_cfg=model_cfg,
                model_name=model_name,
                task_cfg=task_cfg,
                runtime_cfg=runtime_cfg,
                base_url=handle.base_url,
            )
            # Compute metrics immediately (eager per-model)
            try:
                systems_path, cognitive_path = compute_metrics_for_model(
                    model_name=model_name,
                    raw_result_path=raw_path,
                    task_cfg=task_cfg,
                    systems_profile=systems_profile,
                    cognitive_profile=cognitive_profile,
                )

                summary["models"].append(
                    {
                        "model_name": model_name,
                        "raw_results": str(raw_path),
                        "serve_mode": handle.mode,
                        "attempted_modes": handle.attempted_modes,
                        "systems_summary": str(systems_path),
                        "cognitive_summary": str(cognitive_path) if cognitive_path else None,
                    }
                )
            except Exception as exc_inner:
                summary["models"].append(
                    {
                        "model_name": model_name,
                        "raw_results": str(raw_path),
                        "serve_mode": handle.mode,
                        "attempted_modes": handle.attempted_modes,
                        "error": f"metrics_error: {exc_inner}",
                    }
                )
                print(f"  ERROR computing metrics for {model_name}: {exc_inner}")
        except Exception as exc:
            summary["models"].append(
                {
                    "model_name": model_name,
                    "error": str(exc),
                }
            )
            print(f"  ERROR: {exc}")
        finally:
            stop_vllm_server(handle)

    # Write orchestration summary
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "multi_model_orchestrator_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failed = [m for m in summary["models"] if "error" in m]
    print("\\n" + "=" * 80)
    print("Run complete")
    print(f"Summary: {summary_path}")
    print(f"Successful models: {len(summary['models']) - len(failed)}")
    print(f"Failed models: {len(failed)}")
    print("=" * 80)

    if failed:
        sys.exit(1)

    for model_entry in summary["models"]:
        model_name = model_entry["model_name"]
        if "error" in model_entry:
            continue

        raw_path = raw_result_map.get(model_name)
        if raw_path is None:
            model_entry["error"] = "Missing raw results path"
            continue

        try:
            systems_path, cognitive_path = compute_metrics_for_model(
                model_name=model_name,
                raw_result_path=raw_path,
                task_cfg=task_cfg,
                systems_profile=systems_profile,
                cognitive_profile=cognitive_profile,
            )
            model_entry["systems_summary"] = str(systems_path)
            model_entry["cognitive_summary"] = str(cognitive_path) if cognitive_path else None
            print(f"  Metrics for {model_name}: {systems_path}")
        except Exception as exc:
            model_entry["error"] = str(exc)
            print(f"  ERROR computing metrics for {model_name}: {exc}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "multi_model_orchestrator_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    failed = [m for m in summary["models"] if "error" in m]
    print("\\n" + "=" * 80)
    print("Run complete")
    print(f"Summary: {summary_path}")
    print(f"Successful models: {len(summary['models']) - len(failed)}")
    print(f"Failed models: {len(failed)}")
    print("=" * 80)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
