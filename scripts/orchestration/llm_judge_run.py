from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from scripts.vllm.vllm_manager import start_vllm, stop_vllm
from scripts.orchestration.orchestrator import (
    RAW_RESULTS_DIR,
    REPORTS_DIR,
    SERVE_LOG_DIR,
    discover_task_configs,
    load_dotenv,
    load_yaml,
)

from llm_bench.judge import JudgeClient, build_judge_messages, build_judge_response_format, parse_judge_response
from llm_bench.metrics import summarize_judge_group
from llm_bench.schemas import BenchmarkResult

JUDGED_RESULTS_DIR = Path("outputs/judged")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline LLM-as-judge scoring pass over saved raw results.")
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Path to a judge model config, e.g. configs/models/judges/medgemma_27b_it_judge.yaml",
    )
    parser.add_argument("--serve-port", type=int, default=int(os.getenv("JUDGE_SERVE_PORT", "8010")))
    parser.add_argument(
        "--startup-timeout-s",
        type=int,
        default=int(os.getenv("VLLM_STARTUP_TIMEOUT_S", "1800")),
    )
    return parser.parse_args()


def run_task(judge_client: JudgeClient, task_cfg_path: Path) -> None:
    task_cfg = load_yaml(task_cfg_path)
    task_id = task_cfg["task_id"]

    cognitive_profile_path = task_cfg.get("metrics", {}).get("cognitive_profile")
    if not cognitive_profile_path:
        return
    cognitive_profile = load_yaml(cognitive_profile_path)

    judge_cfg_block = cognitive_profile.get("llm_judge", {})
    if not judge_cfg_block.get("enabled", False):
        return

    prompt_cfg = load_yaml(judge_cfg_block["judge_prompt_config"])
    rubric = judge_cfg_block.get("rubric", {})
    response_format = build_judge_response_format(task_id, rubric)

    raw_task_dir = RAW_RESULTS_DIR / task_id
    if not raw_task_dir.exists():
        print(f"  No raw results for task {task_id} at {raw_task_dir}, skipping.")
        return

    for raw_path in sorted(raw_task_dir.glob("*.json")):
        model_name = raw_path.stem
        print(f"\n== Judging {task_id}/{model_name} ==")

        with open(raw_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        judged_by_dataset: dict[str, list[dict[str, Any]]] = {}
        for dataset_name, items in raw_data.items():
            rows = []
            for item in tqdm(items, desc=f"{task_id}/{model_name}/{dataset_name}"):
                result = BenchmarkResult(**item)
                messages = build_judge_messages(result, prompt_cfg)
                raw_text = judge_client.complete(messages, response_format)
                parsed = parse_judge_response(raw_text, rubric)
                rows.append(
                    {
                        "sample_id": result.sample_id,
                        "scores": parsed or {},
                        "parse_ok": parsed is not None,
                    }
                )
            judged_by_dataset[dataset_name] = rows

        judged_path = JUDGED_RESULTS_DIR / task_id / f"{model_name}.json"
        judged_path.parent.mkdir(parents=True, exist_ok=True)
        judged_path.write_text(json.dumps(judged_by_dataset, indent=2), encoding="utf-8")
        print(f"  Judged results: {judged_path}")

        summary_path = REPORTS_DIR / task_id / model_name / "cognitive_summary.json"
        if not summary_path.exists():
            print(f"  WARNING: no cognitive_summary.json at {summary_path}, skipping merge.")
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        for group in summary.get("groups", []):
            rows = judged_by_dataset.get(group["dataset"], [])
            group.setdefault("metrics", {})["llm_judge"] = summarize_judge_group(rows, rubric)

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  Cognitive summary updated: {summary_path}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    judge_cfg = load_yaml(args.judge_model)
    startup_timeout_s = int(judge_cfg.get("startup_timeout_s", args.startup_timeout_s))

    handle = start_vllm(
        model_cfg=judge_cfg,
        port=args.serve_port,
        logs_dir=SERVE_LOG_DIR,
        timeout_s=startup_timeout_s,
    )
    print(f"Judge serving at {handle.base_url} (mode={handle.mode})")

    try:
        judge_client = JudgeClient(
            base_url=handle.base_url,
            model_id=str(judge_cfg["model_id"]),
            api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        )

        for task_cfg_path in discover_task_configs():
            print(f"\n{'='*80}\nTASK: {task_cfg_path.stem}\n{'='*80}")
            run_task(judge_client, task_cfg_path)
    finally:
        stop_vllm(handle)


if __name__ == "__main__":
    main()
