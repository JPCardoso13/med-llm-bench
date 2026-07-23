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


def _extract_core_metrics(mcq: dict[str, Any] | None, generative_metrics: dict[str, Any] | None) -> dict[str, float]:
    """Shared extraction from an already-located mcq dict and/or generative metrics dict."""
    out: dict[str, float] = {}

    if mcq:
        for key in ("accuracy", "precision", "recall"):
            value = _extract_scalar(mcq.get(key, {}), key)
            if value is not None:
                out[key] = value

    if generative_metrics:
        token_f1 = _extract_scalar(generative_metrics.get("answer", {}).get("token_f1", {}), "token_f1")
        if token_f1 is not None:
            out["token_f1"] = token_f1
        for rouge_key, rouge_node in generative_metrics.get("rouge", {}).items():
            value = _extract_scalar(rouge_node, rouge_key)
            if value is not None:
                out[rouge_key] = value

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

    fields: if given, only these group_field names are kept (matched by name,
    across whichever datasets happen to expose them) - not every grouping
    field a dataset exposes is equally useful for a headline comparison, and
    this function stays generic rather than hardcoding which ones matter for
    any particular project. The actual list is the caller's choice (see
    GROUP_BY_FIELDS in scripts/analysis/generate_report.py), not baked in here.
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


def load_systems_metrics(reports_dir: str | Path = "outputs/reports") -> pd.DataFrame:
    """One row per (task, model, dataset, metric_name) from systems_summary.json.

    Pulls latency (mean and p99 - the tail, not just the average, since a
    model that's fast on average but has a long tail is a different risk
    profile than a consistently fast one) and throughput means. The full
    systems_summary.json also has p50/p90/p95 and telemetry that aren't
    needed for the tradeoffs/ charts yet.
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

                total_latency = metrics.get("latency", {}).get("total_latency_ms", {})
                if total_latency.get("mean") is not None:
                    rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                  "metric_name": "total_latency_ms", "value": total_latency["mean"]})
                if total_latency.get("p99") is not None:
                    rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                  "metric_name": "total_latency_ms_p99", "value": total_latency["p99"]})

                for throughput_name, throughput_node in metrics.get("throughput", {}).items():
                    mean = throughput_node.get("mean")
                    if mean is not None:
                        rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                      "metric_name": throughput_name, "value": mean})

    return pd.DataFrame(rows, columns=["task_id", "model_name", "dataset", "metric_name", "value"])


def load_reliability_metrics(reports_dir: str | Path = "outputs/reports") -> pd.DataFrame:
    """One row per (task, model, dataset, metric_name) - failure/quality rates, not scores.

    Reads the "quality" block already computed alongside each dataset group
    (parse_failure_count, ambiguous_extraction_count, missing_ref_field_count,
    truncated_count), normalized to a rate using that group's sample_count.
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
                for count_key in ("parse_failure_count", "ambiguous_extraction_count", "missing_ref_field_count", "truncated_count"):
                    count = quality.get(count_key)
                    if count is None:
                        continue
                    rate_name = count_key.replace("_count", "_rate")
                    rows.append({"task_id": task_id, "model_name": model_name, "dataset": dataset,
                                  "metric_name": rate_name, "value": count / sample_count})

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
