from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _extract_scalar(node: dict[str, Any], key: str) -> float | None:
    """Pull a single representative number out of one metric's sub-dict.

    Metric shapes in cognitive_summary.json aren't uniform (accuracy is
    self-named: {"accuracy": {"accuracy": 0.5, ...}}; token_f1/rouge use
    {"mean": ...}; precision/recall use {"value": ...}) - covering these
    three conventions generically avoids hardcoding per-task-type metric
    names here, matching how the calculators themselves stay config-driven.
    """
    if not isinstance(node, dict):
        return node if isinstance(node, (int, float)) else None
    for candidate_key in (key, "mean", "value"):
        value = node.get(candidate_key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _extract_answer_metrics(answer_metrics: dict[str, Any]) -> dict[str, float]:
    """Generic extraction from an "answer" target's similarity-metrics
    sub-dict - {metric_name: score_or_{variant: score}} - shared by both mcq
    and generative profiles (cognitive_calculator.py wires the same
    similarity.metrics mechanism into both). Iterates whatever's actually
    present instead of a hardcoded metric-name list (previously just
    token_f1/rouge - exact_match was silently never extracted here despite
    being configurable), so a newly configured metric like bertscore surfaces
    without a matching update in this file.

    A multi-valued metric's node is all-dict-valued (rouge: {"rouge1": {...},
    ...}; bertscore: {"precision": {...}, ...}); flattens to bare variant
    keys, same convention rouge already used - fine as long as two
    multi-valued metrics configured on the same target don't share a variant
    name (rouge's rougeN names and bertscore's precision/recall/f1 don't).
    """
    out: dict[str, float] = {}
    for metric_name, metric_node in answer_metrics.items():
        if isinstance(metric_node, dict) and metric_node and all(isinstance(v, dict) for v in metric_node.values()):
            for variant_key, variant_node in metric_node.items():
                value = _extract_scalar(variant_node, variant_key)
                if value is not None:
                    out[variant_key] = value
        else:
            value = _extract_scalar(metric_node, metric_name)
            if value is not None:
                out[metric_name] = value
    return out


def _extract_core_metrics(mcq: dict[str, Any] | None, generative_metrics: dict[str, Any] | None) -> dict[str, float]:
    """Shared extraction from an already-located mcq dict and/or generative metrics dict."""
    out: dict[str, float] = {}

    if mcq:
        for key in ("accuracy", "precision", "recall", "f1"):
            value = _extract_scalar(mcq.get(key, {}), key)
            if value is not None:
                out[key] = value
        out.update(_extract_answer_metrics(mcq.get("answer", {})))

    if generative_metrics:
        out.update(_extract_answer_metrics(generative_metrics.get("answer", {})))
        compression_ratio = _extract_scalar(generative_metrics.get("compression_ratio", {}), "compression_ratio")
        if compression_ratio is not None:
            out["compression_ratio"] = compression_ratio

    return out


def _headline_metrics_from_group(metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten one top-level dataset group's metrics.mcq / metrics.generative.metrics block."""
    mcq = metrics.get("mcq")
    generative_metrics = metrics.get("generative", {}).get("metrics")
    return _extract_core_metrics(mcq, generative_metrics)


def _headline_metrics_from_group_by_bucket(bucket: dict[str, Any]) -> dict[str, float]:
    """Flatten one group_by bucket - a DIFFERENT shape from the top-level group.

    cognitive_calculator.py's group_by reuses the same per-task-type summarizer
    output directly, without the top-level's extra "mcq"/"generative" wrapper:
    MCQ buckets have accuracy/precision/recall as top-level keys; generative
    buckets have a "metrics" key (not "generative.metrics") wrapping
    answer/rouge. Both shapes are handled here rather than assumed identical
    to the top-level group - confirmed by inspecting real output, not assumed.
    """
    if "accuracy" in bucket:
        return _extract_core_metrics(bucket, None)
    if "metrics" in bucket:
        return _extract_core_metrics(None, bucket["metrics"])
    return {}


def load_headline_metrics(reports_dir: str | Path = "outputs/reports") -> pd.DataFrame:
    """One row per (task, model, dataset, metric_name) - the top-level, non-group_by numbers.

    Discovers task/model pairs by walking outputs/reports/<task_id>/<model_name>/
    cognitive_summary.json - no hardcoded task or model list, so this picks up
    whatever a given run actually produced.
    """
    rows: list[dict[str, Any]] = []
    reports_dir = Path(reports_dir)

    for task_dir in sorted(reports_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        for model_dir in sorted(task_dir.glob("*")):
            summary_path = model_dir / "cognitive_summary.json"
            if not summary_path.exists():
                continue
            model_name = model_dir.name

            data = json.loads(summary_path.read_text())
            for group in data.get("groups", []):
                dataset = group["dataset"]
                for metric_name, value in _headline_metrics_from_group(group["metrics"]).items():
                    rows.append({
                        "task_id": task_id,
                        "model_name": model_name,
                        "dataset": dataset,
                        "metric_name": metric_name,
                        "value": value,
                    })

    return pd.DataFrame(rows, columns=["task_id", "model_name", "dataset", "metric_name", "value"])


def load_group_by_metrics(reports_dir: str | Path = "outputs/reports", fields: list[str] | None = None) -> pd.DataFrame:
    """One row per (task, model, dataset, group_field, group_value, metric_name).

    Reads the group_by block cognitive_calculator.py already computes per
    dataset group - same metric-extraction convention as load_headline_metrics,
    reused rather than duplicated.

    fields: optional filter, kept for flexibility - unused by default. Which
    fields are worth grouping by is decided once at the source (each dataset
    config's mapping.grouping vs mapping.metadata), not curated here or in
    the caller, so callers normally pass nothing and this returns whatever
    the calculator actually found.
    """
    rows: list[dict[str, Any]] = []
    reports_dir = Path(reports_dir)
    field_filter = set(fields) if fields is not None else None

    for task_dir in sorted(reports_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        for model_dir in sorted(task_dir.glob("*")):
            summary_path = model_dir / "cognitive_summary.json"
            if not summary_path.exists():
                continue
            model_name = model_dir.name

            data = json.loads(summary_path.read_text())
            for group in data.get("groups", []):
                dataset = group["dataset"]
                for group_field, buckets in group.get("group_by", {}).items():
                    if field_filter is not None and group_field not in field_filter:
                        continue
                    for group_value, bucket_summary in buckets.items():
                        metrics = _headline_metrics_from_group_by_bucket(bucket_summary)
                        for metric_name, value in metrics.items():
                            rows.append({
                                "task_id": task_id,
                                "model_name": model_name,
                                "dataset": dataset,
                                "group_field": group_field,
                                "group_value": group_value,
                                "metric_name": metric_name,
                                "value": value,
                                "sample_count": bucket_summary.get("sample_count"),
                            })

    return pd.DataFrame(
        rows,
        columns=["task_id", "model_name", "dataset", "group_field", "group_value", "metric_name", "value", "sample_count"],
    )


def load_mcq_label_metrics(reports_dir: str | Path = "outputs/reports") -> pd.DataFrame:
    """Per-answer-letter precision/recall/f1 from _build_mcq_classification_summary's
    per_label breakdown (cognitive_calculator.py) - computed since this
    project's early sessions but never surfaced anywhere before now; only
    the macro-averaged scalar reached reports.

    Shaped identically to load_group_by_metrics's output (group_field fixed
    to "answer_letter") specifically so this can be concatenated straight
    into that DataFrame and reuse build_subgroups/plot_group_by_heatmap/
    pivot_group_by_table as-is - answer letter is just one more grouping
    dimension, no new chart/table code needed.
    """
    rows: list[dict[str, Any]] = []
    reports_dir = Path(reports_dir)

    for task_dir in sorted(reports_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        for model_dir in sorted(task_dir.glob("*")):
            summary_path = model_dir / "cognitive_summary.json"
            if not summary_path.exists():
                continue
            model_name = model_dir.name

            data = json.loads(summary_path.read_text())
            for group in data.get("groups", []):
                dataset = group["dataset"]
                mcq = group["metrics"].get("mcq")
                if not mcq:
                    continue
                for metric_name in ("precision", "recall", "f1"):
                    per_label = mcq.get(metric_name, {}).get("per_label", {})
                    for label, stats in per_label.items():
                        value = stats.get(metric_name)
                        if value is None:
                            continue
                        tp = stats.get("tp", 0) or 0
                        fn = stats.get("fn", 0) or 0
                        rows.append({
                            "task_id": task_id,
                            "model_name": model_name,
                            "dataset": dataset,
                            "group_field": "answer_letter",
                            "group_value": label,
                            "metric_name": metric_name,
                            "value": value,
                            "sample_count": tp + fn,  # samples whose reference answer was this letter
                        })

    return pd.DataFrame(
        rows,
        columns=["task_id", "model_name", "dataset", "group_field", "group_value", "metric_name", "value", "sample_count"],
    )


def load_judge_distributions(reports_dir: str | Path = "outputs/reports") -> pd.DataFrame:
    """One row per (task, model, dataset, rubric_item, label, percentage).

    Only present for tasks with a judge wired up (OECR/SRC) - CDKR simply
    won't appear here, by the same judge-by-design absence already reflected
    in cognitive_summary.json.
    """
    rows: list[dict[str, Any]] = []
    reports_dir = Path(reports_dir)

    for task_dir in sorted(reports_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        for model_dir in sorted(task_dir.glob("*")):
            summary_path = model_dir / "cognitive_summary.json"
            if not summary_path.exists():
                continue
            model_name = model_dir.name

            data = json.loads(summary_path.read_text())
            for group in data.get("groups", []):
                dataset = group["dataset"]
                llm_judge = group["metrics"].get("llm_judge")
                if not llm_judge:
                    continue
                for rubric_item, item_summary in llm_judge.get("items", {}).items():
                    labels = item_summary.get("labels", [])
                    percentages = item_summary.get("percentages", {})
                    for label in labels:
                        rows.append({
                            "task_id": task_id,
                            "model_name": model_name,
                            "dataset": dataset,
                            "rubric_item": rubric_item,
                            "label": label,
                            "label_order": labels.index(label),
                            "percentage": percentages.get(label, 0.0) or 0.0,
                        })

    return pd.DataFrame(
        rows,
        columns=["task_id", "model_name", "dataset", "rubric_item", "label", "label_order", "percentage"],
    )


def load_judge_agreement(reports_dir: str | Path = "outputs/reports") -> pd.DataFrame:
    """One row per (task, model, dataset, metric_name, value) - Spearman
    correlation between one rubric item's judge verdict and one automatic
    metric's per-sample score (cognitive_calculator.py::summarize_judge_agreement,
    written into cognitive_summary.json's llm_judge.agreement block by
    llm_judge_run.py). Answers "does this automatic metric actually track
    what the judge says" - previously computed but never surfaced anywhere.

    metric_name is "<rubric_item>_<automatic_metric>" (e.g.
    "diagnosis_correctness_token_f1") so this can reuse
    plot_headline_bar_chart/plot_grouped_metric_bar_chart/pivot_headline_table
    directly - same (task_id, model_name, dataset, metric_name, value) shape
    as load_headline_metrics, not a new chart/table shape to build.
    """
    rows: list[dict[str, Any]] = []
    reports_dir = Path(reports_dir)

    for task_dir in sorted(reports_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        for model_dir in sorted(task_dir.glob("*")):
            summary_path = model_dir / "cognitive_summary.json"
            if not summary_path.exists():
                continue
            model_name = model_dir.name

            data = json.loads(summary_path.read_text())
            for group in data.get("groups", []):
                dataset = group["dataset"]
                agreement = group["metrics"].get("llm_judge", {}).get("agreement", {})
                for rubric_item, item_agreement in agreement.items():
                    for metric_name, stats in item_agreement.items():
                        spearman = stats.get("spearman")
                        if spearman is not None:
                            rows.append({
                                "task_id": task_id,
                                "model_name": model_name,
                                "dataset": dataset,
                                "metric_name": f"{rubric_item}_{metric_name}",
                                "value": spearman,
                            })

    return pd.DataFrame(rows, columns=["task_id", "model_name", "dataset", "metric_name", "value"])


def load_systems_metrics(reports_dir: str | Path = "outputs/reports") -> pd.DataFrame:
    """One row per (task, model, dataset, metric_name) from systems_summary.json.

    Pulls every field under `latency:` (mean, p99 - the tail, not just the
    average, since a model that's fast on average but has a long tail is a
    different risk profile than a consistently fast one - and cov, mean/std's
    ratio) - whatever fields configs/metrics/systems.yaml's latency.fields
    lists (today: total_latency_ms, ttft_ms), not a hardcoded pair - and
    throughput means. The full systems_summary.json also has p50/p90/p95,
    inter_token_latency, usage (token counts), generation (perplexity), and
    telemetry (GPU util/memory) that aren't pulled here yet.
    """
    rows: list[dict[str, Any]] = []
    reports_dir = Path(reports_dir)

    for task_dir in sorted(reports_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        for model_dir in sorted(task_dir.glob("*")):
            summary_path = model_dir / "systems_summary.json"
            if not summary_path.exists():
                continue
            model_name = model_dir.name

            data = json.loads(summary_path.read_text())
            for group in data.get("groups", []):
                dataset = group["dataset"]
                metrics = group.get("metrics", {})

                for latency_field, latency_node in metrics.get("latency", {}).items():
                    if latency_node.get("mean") is not None:
                        rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                      "metric_name": latency_field, "value": latency_node["mean"]})
                    if latency_node.get("p99") is not None:
                        rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                      "metric_name": f"{latency_field}_p99", "value": latency_node["p99"]})
                    # Coefficient of variation (std/mean) - a scale-free measure of
                    # how spread out a latency field is, so a slow-but-consistent
                    # model and a fast-but-erratic one don't look the same just
                    # because their means happen to be close. Derived here from
                    # mean+std already computed by the calculator, not a new
                    # calculator metric.
                    mean = latency_node.get("mean")
                    std = latency_node.get("std")
                    if mean is not None and std is not None and mean > 0:
                        rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                      "metric_name": f"{latency_field}_cov", "value": std / mean})

                for throughput_name, throughput_node in metrics.get("throughput", {}).items():
                    mean = throughput_node.get("mean")
                    if mean is not None:
                        rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                      "metric_name": throughput_name, "value": mean})

    return pd.DataFrame(rows, columns=["task_id", "model_name", "dataset", "metric_name", "value"])


def load_reliability_metrics(
    reports_dir: str | Path = "outputs/reports",
    judge_flag_rates: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """One row per (task, model, dataset, metric_name) - failure/quality rates, not scores.

    Reads the "quality" block already computed alongside each dataset group
    (parse_failure_count, ambiguous_extraction_count, missing_ref_field_count,
    truncated_count, shrunk_budget_truncated_count), normalized to a rate
    using that group's sample_count.

    judge_flag_rates: optional (rubric_item, label) pairs whose label
    percentage should also be surfaced here as a "bad outcome rate" - e.g.
    [("safety_flag", "Unsafe")]. Deliberately not hardcoded in this function:
    the judge rubric is fully generic (arbitrary item names, arbitrary label
    sets, no consistent "worst label" position), so which pairs count as a
    reliability concern is a project-specific curation choice - see
    JUDGE_FLAG_RATES in scripts/analysis/generate_report.py, same pattern as
    TRADEOFF_QUALITY_METRIC.
    """
    rows: list[dict[str, Any]] = []
    reports_dir = Path(reports_dir)

    for task_dir in sorted(reports_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        for model_dir in sorted(task_dir.glob("*")):
            summary_path = model_dir / "cognitive_summary.json"
            if not summary_path.exists():
                continue
            model_name = model_dir.name

            data = json.loads(summary_path.read_text())
            for group in data.get("groups", []):
                dataset = group["dataset"]
                sample_count = group.get("sample_count") or 0
                quality = group.get("quality", {})
                if sample_count <= 0:
                    continue
                for count_key in ("parse_failure_count", "ambiguous_extraction_count", "missing_ref_field_count", "truncated_count", "shrunk_budget_truncated_count"):
                    count = quality.get(count_key)
                    if count is None:
                        continue
                    rate_name = count_key.replace("_count", "_rate")
                    rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                  "metric_name": rate_name, "value": count / sample_count})

                # Model bleeding a second section into its answer (e.g. an
                # unlabeled "Explanation:" appended after the actual answer)
                # - a distinct failure mode from parse_failure_rate (which
                # only sees content, not extraction success), computed since
                # early in this project but never surfaced until now.
                answer_contaminated_count = (
                    group["metrics"].get("generative", {}).get("metrics", {}).get("format", {}).get("answer_contaminated_count")
                )
                if answer_contaminated_count is not None:
                    rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                  "metric_name": "answer_contaminated_rate", "value": answer_contaminated_count / sample_count})

                llm_judge = group["metrics"].get("llm_judge")
                if llm_judge and judge_flag_rates:
                    items = llm_judge.get("items", {})
                    for rubric_item, label in judge_flag_rates:
                        item_summary = items.get(rubric_item)
                        if not item_summary:
                            continue
                        percentage = item_summary.get("percentages", {}).get(label)
                        if percentage is None:
                            continue
                        rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                      "metric_name": f"{rubric_item}_{label.lower()}_rate", "value": percentage})

    return pd.DataFrame(rows, columns=["task_id", "model_name", "dataset", "metric_name", "value"])


def load_qualitative_examples(reports_dir: str | Path = "outputs/reports", top_n: int = 2) -> pd.DataFrame:
    """Best/worst-scoring sample previews per (task, model, dataset) - top_n each side.

    Pulled from the diagnostics.worst_samples/best_samples the calculator
    already curates (already sorted, already truncated to preview length) -
    this just picks the top_n and flattens them for a table, no new scoring.
    """
    rows: list[dict[str, Any]] = []
    reports_dir = Path(reports_dir)

    for task_dir in sorted(reports_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        for model_dir in sorted(task_dir.glob("*")):
            summary_path = model_dir / "cognitive_summary.json"
            if not summary_path.exists():
                continue
            model_name = model_dir.name

            data = json.loads(summary_path.read_text())
            for group in data.get("groups", []):
                dataset = group["dataset"]
                diagnostics = group["metrics"].get("generative", {}).get("diagnostics")
                if not diagnostics:
                    continue
                for rank_label, samples in (("worst", diagnostics.get("worst_samples", [])), ("best", diagnostics.get("best_samples", []))):
                    for sample in samples[:top_n]:
                        rows.append({
                            "task_id": task_id,
                            "model_name": model_name,
                            "dataset": dataset,
                            "rank": rank_label,
                            "sample_id": sample.get("sample_id"),
                            "composite_score": sample.get("composite_score"),
                            "response_preview": sample.get("response_preview"),
                            "reference_answer": sample.get("reference_answer"),
                        })

    return pd.DataFrame(
        rows,
        columns=["task_id", "model_name", "dataset", "rank", "sample_id", "composite_score", "response_preview", "reference_answer"],
    )


def load_label_bias(reports_dir: str | Path = "outputs/reports") -> pd.DataFrame:
    """One row per (task, model, dataset, letter, kind, value) from MCQ's
    label_distribution block - kind is "predicted" or "reference", value is
    that letter's share (0-1) of that kind's total.
    """
    rows: list[dict[str, Any]] = []
    reports_dir = Path(reports_dir)

    for task_dir in sorted(reports_dir.glob("*")):
        if not task_dir.is_dir():
            continue
        task_id = task_dir.name

        for model_dir in sorted(task_dir.glob("*")):
            summary_path = model_dir / "cognitive_summary.json"
            if not summary_path.exists():
                continue
            model_name = model_dir.name

            data = json.loads(summary_path.read_text())
            for group in data.get("groups", []):
                dataset = group["dataset"]
                distribution = group["metrics"].get("mcq", {}).get("label_distribution")
                if not distribution:
                    continue
                for kind in ("predicted", "reference"):
                    for letter, pct in distribution.get(f"{kind}_pct", {}).items():
                        if pct is None:
                            continue
                        rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                      "letter": letter, "kind": kind, "value": pct})

    return pd.DataFrame(rows, columns=["task_id", "model_name", "dataset", "letter", "kind", "value"])
