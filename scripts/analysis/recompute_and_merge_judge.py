"""Recompute a task's metrics from raw/ AND merge already-saved judged/
scores into the result - for recovering a task whose generation and judge
pass both finished (raw/ and judged/ both exist) but whose Pass 2 metrics
computation never ran (e.g. the run timed out on a later task before
orchestrator.py ever reached Pass 2). Without this, a completed judge pass
has nothing to merge into (see llm_judge_run.py's "no cognitive_summary.json
... skipping merge" warning) and its judged/ output sits unused.

Usage: RUN_ID=<id> python3 scripts/analysis/recompute_and_merge_judge.py <task_id> <model_name>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.orchestration.orchestrator import (
    OUTPUTS_DIR,
    RAW_RESULTS_DIR,
    REPORTS_DIR,
    TASKS_DIR,
    compute_and_save_metrics,
    load_task_bundle,
)
from llm_bench.metrics import summarize_judge_agreement, summarize_judge_group

JUDGED_RESULTS_DIR = OUTPUTS_DIR / "judged"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: recompute_and_merge_judge.py <task_id> <model_name>")

    task_id, model_name = sys.argv[1], sys.argv[2]

    bundle = load_task_bundle(TASKS_DIR / f"{task_id}.yaml")
    raw_path = RAW_RESULTS_DIR / task_id / f"{model_name}.json"
    if not raw_path.exists():
        raise SystemExit(f"no raw file at {raw_path}")

    _, cognitive_path = compute_and_save_metrics(
        model_name=model_name,
        raw_path=raw_path,
        task_id=task_id,
        systems_profile=bundle["systems_profile"],
        cognitive_profile=bundle["cognitive_profile"],
    )
    print(f"  Recomputed: {cognitive_path}")

    judged_path = JUDGED_RESULTS_DIR / task_id / f"{model_name}.json"
    if not judged_path.exists():
        print(f"  No judged/ file at {judged_path} - nothing to merge, done.")
        return

    judged_by_dataset = json.loads(judged_path.read_text())
    cognitive_profile = bundle["cognitive_profile"]
    judge_cfg_block = cognitive_profile.get("llm_judge", {})
    rubric = judge_cfg_block.get("rubric", {})

    summary_path = REPORTS_DIR / task_id / model_name / "cognitive_summary.json"
    summary = json.loads(summary_path.read_text())

    for group in summary.get("groups", []):
        rows = judged_by_dataset.get(group["dataset"], [])
        if not rows:
            continue
        group.setdefault("metrics", {})["llm_judge"] = summarize_judge_group(rows, rubric)
        per_sample_scores = group["metrics"].get("generative", {}).get("per_sample_scores")
        if per_sample_scores:
            group["metrics"]["llm_judge"]["agreement"] = summarize_judge_agreement(rows, per_sample_scores, rubric)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Merged judged/ scores into: {summary_path}")


if __name__ == "__main__":
    main()
