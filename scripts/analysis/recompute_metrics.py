"""Recompute a task's metrics for existing raw/ output without re-running
generation - used when a job dies before orchestrator.py's Pass 2 (metrics
computation runs only after ALL tasks finish generating, so a TIMEOUT/crash
mid-generation on a later task skips Pass 2 entirely, even for earlier
tasks whose raw/ output is complete). Originally split off from
recompute_cdkr_metrics.py (2026-09-04, MCQ regex fix) to be task-agnostic
after finding the same gap on RUN_ID=175482's oecr (raw complete, but
reports/oecr never existed because 175482 timed out mid-src, before Pass 2
ever ran).

Usage: RUN_ID=<id> python3 scripts/analysis/recompute_metrics.py <task_id> <model_name> [<model_name> ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.orchestration.orchestrator import (
    RAW_RESULTS_DIR,
    TASKS_DIR,
    compute_and_save_metrics,
    load_task_bundle,
)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: recompute_metrics.py <task_id> <model_name> [<model_name> ...]")

    task_id = sys.argv[1]
    model_names = sys.argv[2:]

    bundle = load_task_bundle(TASKS_DIR / f"{task_id}.yaml")

    for model_name in model_names:
        raw_path = RAW_RESULTS_DIR / task_id / f"{model_name}.json"
        if not raw_path.exists():
            print(f"  SKIP {model_name}: no raw file at {raw_path}")
            continue
        systems_path, cognitive_path = compute_and_save_metrics(
            model_name=model_name,
            raw_path=raw_path,
            task_id=task_id,
            systems_profile=bundle["systems_profile"],
            cognitive_profile=bundle["cognitive_profile"],
        )
        print(f"  {model_name}: rewrote {systems_path} and {cognitive_path}")


if __name__ == "__main__":
    main()
