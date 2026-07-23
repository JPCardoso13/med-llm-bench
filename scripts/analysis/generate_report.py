from __future__ import annotations

import argparse
from pathlib import Path

from llm_bench.reporting import (
    load_headline_metrics,
    load_group_by_metrics,
    load_judge_distributions,
    load_systems_metrics,
    load_reliability_metrics,
    load_qualitative_examples,
    plot_headline_bar_chart,
    plot_grouped_metric_bar_chart,
    plot_group_by_heatmap,
    plot_judge_distribution,
    plot_tradeoff_scatter,
    pivot_headline_table,
    pivot_group_by_table,
    pivot_judge_table,
    pivot_reliability_table,
    qualitative_examples_table,
    save_table,
    save_qualitative_examples,
)

REPORTS_DIR = Path("outputs/reports")
ANALYSIS_DIR = Path("outputs/analysis")

# --- Small, editable report-curation choices, kept together here rather
# than scattered or split into their own config files - these are downstream
# display decisions (what to chart), not facts about the data or the
# benchmarking pipeline itself, so they live with the script that consumes
# them. Cheap to change and cheap to reverse: edit a line, re-run this
# script - no re-running the benchmark, no touching dataset/task configs. ---

# The "quality" metric used as the y-axis of every tradeoffs/ scatter, per
# task_type - accuracy for MCQ, token_f1 for generative (present for both
# OECR and SRC, unlike ROUGE which is more summarization-specific).
TRADEOFF_QUALITY_METRIC = {"mcq": "accuracy", "generative": "token_f1"}

# Two x-axis metrics, two different questions: mean decoding throughput asks
# "how fast does it generate" (a rate); p99 total latency asks "how long do
# you actually wait in the worst case" (a tail-risk signal a mean can hide
# entirely - a model fast on average but with a long tail is a different
# risk than a consistently fast one). Both charted, not one replacing the
# other - they're complementary, not redundant.
TRADEOFF_X_METRICS = ["decoding_throughput", "total_latency_ms_p99"]

# MCQ's three headline metrics get combined into one grouped chart instead of
# three separate ones - small enough a set (see the dataviz skill's
# series-count ladder) that color-as-metric stays legible. Generative tasks
# (token_f1/rouge*) are left as separate charts for now, not asked for yet.
MCQ_METRIC_NAMES = {"accuracy", "precision", "recall"}

# Which group_by fields (matched by name, across whichever datasets happen
# to expose them) get charted under subgroups/. Not every grouping field a
# dataset exposes is equally useful for a headline comparison - as of
# 2026-07-22, only body_system is genuinely a body-system/medical-specialty
# field (medxpertqa); everything else in the data today is a different kind
# of axis (exam step, reasoning structure, calculator category, note
# format, document length), not specialty. Add names here to bring any back.
GROUP_BY_FIELDS = ["body_system"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison figures and tables from outputs/reports/.")
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--out-dir", default=str(ANALYSIS_DIR))
    return parser.parse_args()


def _task_dir(out_dir: Path, task_id: str, category: str) -> Path:
    return out_dir / task_id / category


def build_comparison(headline_df, out_dir: Path) -> None:
    for (task_id, dataset), _ in headline_df.groupby(["task_id", "dataset"]):
        cat_dir = _task_dir(out_dir, task_id, "comparison")
        metric_names = list(headline_df[(headline_df["task_id"] == task_id) & (headline_df["dataset"] == dataset)]["metric_name"].unique())

        if set(metric_names) == MCQ_METRIC_NAMES:
            plot_grouped_metric_bar_chart(headline_df, task_id, dataset, sorted(metric_names), cat_dir / f"{dataset}_mcq_metrics.png")
        else:
            for metric_name in metric_names:
                plot_headline_bar_chart(headline_df, task_id, dataset, metric_name, cat_dir / f"{dataset}_{metric_name}.png")

        table = pivot_headline_table(headline_df, task_id, dataset)
        if not table.empty:
            save_table(table, cat_dir, f"{dataset}_headline", caption=f"{task_id}/{dataset}: headline metrics per model")
    print(f"  comparison/ done ({len(headline_df.groupby(['task_id', 'dataset']))} task/dataset pairs)")


def build_subgroups(group_df, out_dir: Path) -> None:
    if group_df.empty:
        print("  subgroups/ skipped (no group_by data - datasets may be too small for any subgroup bucket to be non-empty)")
        return

    keys = group_df[["task_id", "dataset", "group_field", "metric_name"]].drop_duplicates()
    for _, row in keys.iterrows():
        task_id, dataset, group_field, metric_name = row["task_id"], row["dataset"], row["group_field"], row["metric_name"]
        cat_dir = _task_dir(out_dir, task_id, "subgroups")

        plot_group_by_heatmap(group_df, task_id, dataset, group_field, metric_name, cat_dir / f"{dataset}_{group_field}_{metric_name}_heatmap.png")
        table = pivot_group_by_table(group_df, task_id, dataset, group_field, metric_name)
        if not table.empty:
            save_table(table, cat_dir, f"{dataset}_{group_field}_{metric_name}", caption=f"{task_id}/{dataset}: {metric_name} by {group_field}")
    print(f"  subgroups/ done ({len(keys)} heatmaps)")


def build_judge(judge_df, out_dir: Path) -> None:
    if judge_df.empty:
        print("  judge/ skipped (no llm_judge data present - no judge pass run yet, or CDKR which has no judge by design)")
        return

    keys = judge_df[["task_id", "dataset", "rubric_item"]].drop_duplicates()
    for _, row in keys.iterrows():
        task_id, dataset, rubric_item = row["task_id"], row["dataset"], row["rubric_item"]
        cat_dir = _task_dir(out_dir, task_id, "judge")

        plot_judge_distribution(judge_df, task_id, dataset, rubric_item, cat_dir / f"{dataset}_{rubric_item}.png")
        table = pivot_judge_table(judge_df, task_id, dataset, rubric_item)
        if not table.empty:
            save_table(table, cat_dir, f"{dataset}_{rubric_item}", caption=f"{task_id}/{dataset}: judge {rubric_item} label distribution")
    print(f"  judge/ done ({len(keys)} rubric items)")


def build_tradeoffs(systems_df, headline_df, out_dir: Path) -> None:
    count = 0
    for (task_id, dataset), _ in headline_df.groupby(["task_id", "dataset"]):
        available_metrics = set(headline_df[(headline_df["task_id"] == task_id) & (headline_df["dataset"] == dataset)]["metric_name"])
        quality_metric = next((m for m in TRADEOFF_QUALITY_METRIC.values() if m in available_metrics), None)
        if quality_metric is None:
            continue

        cat_dir = _task_dir(out_dir, task_id, "tradeoffs")
        for x_metric in TRADEOFF_X_METRICS:
            try:
                plot_tradeoff_scatter(
                    systems_df, headline_df, task_id, dataset, x_metric, quality_metric,
                    cat_dir / f"{dataset}_{quality_metric}_vs_{x_metric}.png",
                )
                count += 1
            except ValueError:
                continue
    print(f"  tradeoffs/ done ({count} scatter plots)")


def build_reliability(reliability_df, out_dir: Path) -> None:
    if reliability_df.empty:
        print("  reliability/ skipped (no data)")
        return

    count = 0
    for (task_id, dataset), _ in reliability_df.groupby(["task_id", "dataset"]):
        cat_dir = _task_dir(out_dir, task_id, "reliability")
        table = pivot_reliability_table(reliability_df, task_id, dataset)
        if not table.empty:
            save_table(table, cat_dir, f"{dataset}_reliability", caption=f"{task_id}/{dataset}: failure/quality rates per model")
            count += 1
    print(f"  reliability/ done ({count} tables)")


def build_examples(examples_df, out_dir: Path) -> None:
    if examples_df.empty:
        print("  examples/ skipped (no data - MCQ tasks have no free-text response to preview)")
        return

    count = 0
    for (task_id, dataset, model_name), _ in examples_df.groupby(["task_id", "dataset", "model_name"]):
        cat_dir = _task_dir(out_dir, task_id, "examples")
        table = qualitative_examples_table(examples_df, task_id, dataset, model_name)
        if not table.empty:
            save_qualitative_examples(table, cat_dir, f"{dataset}_{model_name}_examples")
            count += 1
    print(f"  examples/ done ({count} tables)")


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)

    headline_df = load_headline_metrics(reports_dir)
    if headline_df.empty:
        print(f"No cognitive_summary.json files found under {reports_dir} - nothing to report.")
        return

    print(f"Group-by fields restricted to: {GROUP_BY_FIELDS}")
    group_df = load_group_by_metrics(reports_dir, fields=GROUP_BY_FIELDS)
    judge_df = load_judge_distributions(reports_dir)
    systems_df = load_systems_metrics(reports_dir)
    reliability_df = load_reliability_metrics(reports_dir)
    examples_df = load_qualitative_examples(reports_dir)

    print(f"Loaded {len(headline_df)} headline / {len(group_df)} group_by / {len(judge_df)} judge / "
          f"{len(systems_df)} systems / {len(reliability_df)} reliability rows, "
          f"{len(examples_df)} example previews.")

    build_comparison(headline_df, out_dir)
    build_subgroups(group_df, out_dir)
    build_judge(judge_df, out_dir)
    build_tradeoffs(systems_df, headline_df, out_dir)
    build_reliability(reliability_df, out_dir)
    build_examples(examples_df, out_dir)

    print(f"\nDone. Output under {out_dir}/<task_id>/<comparison|subgroups|judge|tradeoffs|reliability|examples>/")


if __name__ == "__main__":
    main()
