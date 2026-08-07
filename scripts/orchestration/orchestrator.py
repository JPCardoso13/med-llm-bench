from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from scripts.vllm.vllm_manager import start_vllm, stop_vllm
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
            fewshot_delimiter=prompt_cfg.get("fewshot_delimiter", "\n\n"),
        )

    if task_type == "generative":
        return GenerativeFormatter(
            system_prompt=prompt_cfg["system_prompt"],
            user_turn_template=prompt_cfg["user_turn_template"],
            fewshot_template=prompt_cfg.get("fewshot_template"),
            fewshot_delimiter=prompt_cfg.get("fewshot_delimiter", "\n\n"),
        )

    raise ValueError(f"Unsupported task_type: {task_type}")


def load_run_config() -> dict[str, Any]:
    """Load the run config named by $RUN_CONFIG.

    A run config is an explicit selection ({"tasks": [...], "models": [...]})
    naming what a run covers, by task/model config file stem. Required, not
    optional: configs/models/ is a flat catalog that includes small local-test
    models alongside the 8 production ones, so a silent "nothing set, discover
    everything" fallback would risk quietly running the wrong roster. The
    SLURM wrappers always set RUN_CONFIG (default configs/runs/full_production.yaml)
    - this only bites a direct invocation that forgets to set it, and it
    should bite loudly rather than silently widen scope.
    """
    run_config_path = os.getenv("RUN_CONFIG")
    if not run_config_path:
        raise RuntimeError(
            "RUN_CONFIG is not set. Set it to a configs/runs/*.yaml file explicitly, "
            "e.g. RUN_CONFIG=configs/runs/local_smoke.yaml - there is no implicit "
            "full-catalog fallback, since configs/models/ now includes small "
            "local-test models alongside the 8 production ones."
        )
    return load_yaml(run_config_path)


def _filter_by_selection(configs: list[Path], selected_names: list[str] | None, kind: str) -> list[Path]:
    if not selected_names:
        return configs

    by_stem = {c.stem: c for c in configs}
    missing = [name for name in selected_names if name not in by_stem]
    if missing:
        raise ValueError(
            f"Run config selects unknown {kind}(s) not found in the catalog: {missing}. "
            f"Available: {sorted(by_stem)}"
        )

    return [by_stem[name] for name in selected_names]


def discover_task_configs(run_config: dict[str, Any] | None = None) -> list[Path]:
    """Discover task configs from configs/tasks/*.yaml, sorted by name.

    Narrowed to run_config["tasks"] (matched by file stem) when a run config
    with that key is provided; otherwise every non-template config is used.
    """
    configs = list(TASKS_DIR.glob("*.yaml"))
    configs = [c for c in configs if c.stem != "template"]
    configs = _filter_by_selection(configs, run_config.get("tasks") if run_config else None, "task")
    return sorted(configs)


def discover_model_configs(run_config: dict[str, Any] | None = None) -> list[Path]:
    """Discover model configs from configs/models/*.yaml, sorted by name.

    Narrowed to run_config["models"] (matched by file stem) when a run config
    with that key is provided; otherwise every config in the catalog is used.
    Non-recursive, so configs/models/judges/ is never picked up here - judge
    selection is a separate mechanism (--judge-model / JUDGE_MODEL_CONFIG).
    """
    configs = sorted(MODELS_DIR.glob("*.yaml"))
    return _filter_by_selection(configs, run_config.get("models") if run_config else None, "model")


def run_benchmark_for_model(
    model_cfg: dict[str, Any],
    model_name: str,
    task_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    base_url: str,
    task_id: str,
) -> tuple[Path, dict[str, Any]]:
    default_max_tokens = int(task_cfg.get("execution", {}).get("max_tokens", 1024))
    formatter = build_formatter(task_cfg)
    telemetry_collector = build_telemetry_collector(runtime_cfg)

    num_fewshot = int(task_cfg.get("execution", {}).get("num_fewshot", 0))
    flush_every = int(task_cfg.get("execution", {}).get("flush_every", 10))
    fewshot_seed = task_cfg.get("execution", {}).get("fewshot_seed")
    fewshot_seed = int(fewshot_seed) if fewshot_seed is not None else None
    max_consecutive_failures = int(task_cfg.get("execution", {}).get("max_consecutive_failures", 5))
    task_name = task_cfg.get("task_id", "task")

    datasets_cfg = task_cfg.get("datasets", [])
    enabled_datasets = [d for d in datasets_cfg if d.get("enabled", True)]
    grouped_results: dict[str, list[dict]] = {}
    dataset_status: dict[str, Any] = {}

    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_RESULTS_DIR / task_id / f"{model_name}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    for ds in enabled_datasets:
        dataset_name = ds["name"]
        try:
            dataset_config_path = ds["config"]

            loader = YamlLoader(dataset_config_path)
            data = loader.load()

            eval_limit = ds.get("eval_limit")
            if eval_limit is None:
                eval_limit = PER_DATASET_EVAL_LIMIT

            eval_samples = data.get("eval", [])[: int(eval_limit)]
            fewshot_samples = data.get("fewshot", [])

            # Per-dataset override for the generation length budget, since
            # different datasets in the same task can have very different
            # expected answer lengths (e.g. a long-document summary vs. a short
            # MCQ letter) - falls back to the task-level default when unset.
            max_tokens = int(ds.get("max_tokens", default_max_tokens))
            backend = build_backend(model_cfg, base_url, max_tokens=max_tokens)

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
                fewshot_seed=fewshot_seed,
                max_consecutive_failures=max_consecutive_failures,
            )

            results = runner.run(eval_samples)
            grouped_results[dataset_name] = [r.model_dump(mode="json") for r in results]
            dataset_status[dataset_name] = {
                "attempted": len(eval_samples),
                "succeeded": len(results),
                "error": None,
            }
            print(f"  Dataset {dataset_name}: {len(results)}/{len(eval_samples)} results")
        except Exception as exc:
            # A dataset-level failure (bad dataset config, backend outage,
            # SequentialRunner's circuit breaker tripping on systemic
            # failures, etc.) must not abort sibling datasets for this
            # model/task - skip it and keep going.
            print(f"  ERROR dataset {dataset_name}: {exc}")
            dataset_status[dataset_name] = {"attempted": None, "succeeded": 0, "error": str(exc)}

        # Persist after every dataset attempt, not just once at the end -
        # previously a later dataset's failure would propagate and abort
        # before raw_path was ever written, silently destroying earlier
        # datasets' already-successful results too.
        raw_path.write_text(json.dumps(grouped_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  Raw results: {raw_path}")
    return raw_path, dataset_status


def compute_and_save_metrics(model_name: str, raw_path: Path, task_id: str, systems_profile: dict[str, Any], cognitive_profile: dict[str, Any]) -> tuple[Path, Path | None]:
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Expect grouped-by-dataset raw format: a dict mapping dataset -> list[results]
    if not isinstance(raw_data, dict):
        raise ValueError(f"Expected grouped raw results (dict) in {raw_path}, got {type(raw_data)}")

    flattened: list[dict] = []
    for ds_name, items in raw_data.items():
        if isinstance(items, list):
            flattened.extend(items)

    all_results = [BenchmarkResult(**r) for r in flattened]

    overall_summary = calculate_system_metrics(all_results, systems_profile)

    # Write summaries into per-model directory under task
    task_model_dir = REPORTS_DIR / task_id / model_name
    task_model_dir.mkdir(parents=True, exist_ok=True)

    systems_summary_path = task_model_dir / "systems_summary.json"
    systems_summary_path.write_text(json.dumps(overall_summary, indent=2), encoding="utf-8")

    cognitive_summary_path = None
    if cognitive_profile.get("enabled", True):
        cognitive_overall_summary = calculate_cognitive_metrics(all_results, cognitive_profile)

        cognitive_summary_path = task_model_dir / "cognitive_summary.json"
        cognitive_summary_path.write_text(json.dumps(cognitive_overall_summary, indent=2), encoding="utf-8")

    return systems_summary_path, cognitive_summary_path


def load_task_bundle(task_cfg_path: Path) -> dict[str, Any]:
    """Preload a task's static pieces (profiles, runtime config, id) once.

    None of this depends on which model is currently being served, so it's
    loaded once per task rather than once per (task, model) pair.
    """
    task_cfg = load_yaml(task_cfg_path)
    task_id = task_cfg.get("task_id")
    if not task_id:
        raise ValueError(f"Task config {task_cfg_path} missing 'task_id' field")

    systems_profile = load_yaml(task_cfg["metrics"]["systems_profile"])
    cognitive_profile_path = task_cfg.get("metrics", {}).get("cognitive_profile")
    cognitive_profile = load_yaml(cognitive_profile_path) if cognitive_profile_path else {"enabled": False}
    runtime_cfg = load_yaml(resolve_runtime_config_path(task_cfg))

    return {
        "task_id": task_id,
        "task_cfg": task_cfg,
        "systems_profile": systems_profile,
        "cognitive_profile": cognitive_profile,
        "runtime_cfg": runtime_cfg,
    }


def run_single_model(model_config_path: Path, task_bundles: list[dict[str, Any]], serve_port: int, startup_timeout_s: int) -> dict[str, dict[str, Any]]:
    """Serve one model once and run it across every task.

    Model-outer/task-inner: each model is downloaded and cold-started by
    vLLM exactly once for the whole run, then reused across all tasks,
    instead of the previous task-outer/model-inner nesting which restarted
    (and, under HF_EVICT_BETWEEN_MODELS, re-downloaded) every model once per
    task - Nx redundant loads for an N-task run.
    """
    model_name = model_config_path.stem
    model_cfg = load_yaml(model_config_path)

    print(f"\n== Model: {model_name} ==")
    entries_by_task: dict[str, dict[str, Any]] = {}
    handle = None
    try:
        model_startup_timeout_s = int(model_cfg.get("startup_timeout_s", startup_timeout_s))
        handle = start_vllm(
            model_cfg=model_cfg,
            model_name=model_name,
            port=serve_port,
            logs_dir=SERVE_LOG_DIR,
            timeout_s=model_startup_timeout_s,
        )
        print(f"  Serving at {handle.base_url} (mode={handle.mode})")
        os.environ["LLM_BASE_URL"] = handle.base_url

        for bundle in task_bundles:
            task_id = bundle["task_id"]
            print(f"\n  -- Task: {task_id} --")
            try:
                raw_path, dataset_status = run_benchmark_for_model(
                    model_cfg, model_name, bundle["task_cfg"], bundle["runtime_cfg"], handle.base_url, task_id
                )
                systems_path, cognitive_path = compute_and_save_metrics(
                    model_name, raw_path, task_id, bundle["systems_profile"], bundle["cognitive_profile"]
                )
                print(f"    Systems summary: {systems_path}")
                if cognitive_path:
                    print(f"    Cognitive summary: {cognitive_path}")
                entries_by_task[task_id] = {
                    "model_name": model_name,
                    "raw_results": str(raw_path),
                    "systems_summary": str(systems_path),
                    "cognitive_summary": str(cognitive_path) if cognitive_path else None,
                    "serve_mode": handle.mode,
                    "datasets": dataset_status,
                }
            except Exception as exc:
                print(f"  ERROR processing {model_name} on task {task_id}: {exc}")
                entries_by_task[task_id] = {"model_name": model_name, "error": str(exc)}
    except Exception as exc:
        print(f"ERROR starting {model_name}: {exc}")
        for bundle in task_bundles:
            entries_by_task[bundle["task_id"]] = {"model_name": model_name, "error": str(exc)}
    finally:
        stop_vllm(handle)
        time.sleep(8)
        maybe_evict_hf_cache_between_models()

    return entries_by_task


def main() -> None:
    load_dotenv()

    run_config = load_run_config()
    print(f"Using run config: {os.environ['RUN_CONFIG']}")

    task_configs = discover_task_configs(run_config)
    if not task_configs:
        print("No task configs found in configs/tasks/ (excluding template.yaml)")
        return

    model_configs = discover_model_configs(run_config)
    if not model_configs:
        print("No model configs found in configs/models/")
        return

    serve_port = int(os.getenv("SERVE_PORT", "8000"))
    startup_timeout_s = int(os.getenv("VLLM_STARTUP_TIMEOUT_S", "1800"))

    task_bundles: list[dict[str, Any]] = []
    broken_tasks: list[dict[str, Any]] = []
    for task_cfg_path in task_configs:
        try:
            task_bundles.append(load_task_bundle(task_cfg_path))
        except Exception as exc:
            print(f"ERROR processing task {task_cfg_path.stem}: {exc}")
            broken_tasks.append({"task_id": task_cfg_path.stem, "error": str(exc)})

    task_models: dict[str, list[dict[str, Any]]] = {b["task_id"]: [] for b in task_bundles}

    for model_config_path in model_configs:
        entries_by_task = run_single_model(model_config_path, task_bundles, serve_port, startup_timeout_s)
        for task_id, entry in entries_by_task.items():
            task_models[task_id].append(entry)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = {"tasks": []}
    for bundle in task_bundles:
        task_id = bundle["task_id"]
        task_summary = {"task_id": task_id, "models": task_models[task_id]}
        task_report_dir = REPORTS_DIR / task_id
        task_report_dir.mkdir(parents=True, exist_ok=True)
        summary_path = task_report_dir / "summary.json"
        summary_path.write_text(json.dumps(task_summary, indent=2), encoding="utf-8")
        print(f"\nTask report written to: {summary_path}")
        all_summaries["tasks"].append(task_summary)
    all_summaries["tasks"].extend(broken_tasks)

    # Write overall multi-task summary
    overall_summary_path = REPORTS_DIR / "run_summary.json"
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
