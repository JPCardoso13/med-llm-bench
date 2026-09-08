"""One-off: recompute cdkr metrics for existing raw/ output after the MCQ
answer-extraction regex fix (2026-09-04) - no re-generation needed, this
only re-reads already-saved raw/ JSON and rewrites reports/cdkr/*.

Usage: RUN_ID=<id> python3 scripts/analysis/recompute_cdkr_metrics.py <model_name> [<model_name> ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.orchestration.orchestrator import (
    RAW_RESULTS_DIR,
    compute_and_save_metrics,
    load_task_bundle,
)

TASK_CFG_PATH = Path("configs/tasks/cdkr.yaml")


def main() -> None:
    model_names = sys.argv[1:]
    if not model_names:
        raise SystemExit("usage: recompute_cdkr_metrics.py <model_name> [<model_name> ...]")

    bundle = load_task_bundle(TASK_CFG_PATH)
    task_id = bundle["task_id"]

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
