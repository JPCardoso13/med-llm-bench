from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from llm_bench.metrics.stats import aggregate_values, percentile
from llm_bench.schemas.benchmark_result import BenchmarkResult

def calculate_system_metrics(results: list[BenchmarkResult], profile: Mapping[str, Any]) -> dict[str, Any]:
    if not profile.get("enabled", True):
        return {
            "profile_id": profile.get("profile_id", "systems"),
            "enabled": False,
            "sample_count": len(results),
            "groups": [],
        }

    groups: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        groups[result.dataset].append(result)

    missing_fields: list[dict[str, Any]] = []
    invalid_throughput_samples: set[str] = set()
    group_summaries = []

    for dataset, group_results in groups.items():
        group_summaries.append(
            _summarize_group(
                dataset=dataset,
                results=group_results,
                profile=profile,
                missing_fields=missing_fields,
                invalid_throughput_samples=invalid_throughput_samples,
            )
        )

    quality = _build_quality_summary(
        missing_fields=missing_fields,
        invalid_throughput_samples=invalid_throughput_samples,
        profile=profile,
    )

    fail_on_missing_fields = bool(profile.get("quality_checks", {}).get("fail_on_missing_fields", False))
    if fail_on_missing_fields and quality["missing_field_count"] > 0:
        raise ValueError(
            "System metrics failed: missing fields detected "
            f"({quality['missing_field_count']} samples)."
        )

    return {
        "profile_id": profile.get("profile_id", "systems"),
        "scope": profile.get("scope", "systems"),
        "enabled": True,
        "sample_count": len(results),
        "groups": group_summaries,
        "quality": quality,
    }


def _summarize_group(
    dataset: str,
    results: list[BenchmarkResult],
    profile: Mapping[str, Any],
    missing_fields: list[dict[str, Any]],
    invalid_throughput_samples: set[str],
) -> dict[str, Any]:
    group_summary: dict[str, Any] = {
        "dataset": dataset,
        "sample_count": len(results),
        "metrics": {},
    }

    metrics = group_summary["metrics"]

    latency_cfg = profile.get("latency", {})
    if latency_cfg.get("enabled", False):
        metrics["latency"] = _summarize_direct_fields(
            results=results,
            fields=latency_cfg.get("fields", []),
            aggregates=latency_cfg.get("aggregates", []),
            percentiles=latency_cfg.get("percentiles", []),
            missing_fields=missing_fields,
            dataset=dataset,
        )

    inter_token_cfg = profile.get("inter_token_latency", {})
    if inter_token_cfg.get("enabled", False):
        metrics["inter_token_latency"] = _summarize_inter_token_latency(
            results=results,
            source_field=inter_token_cfg.get("source_field", "inter_token_latencies_ms"),
            request_level_stats=inter_token_cfg.get("request_level_stats", []),
            aggregates=inter_token_cfg.get("aggregates", []),
            percentiles=inter_token_cfg.get("percentiles", []),
            missing_fields=missing_fields,
            dataset=dataset,
        )

    throughput_cfg = profile.get("throughput", {})
    if throughput_cfg.get("enabled", False):
        metrics["throughput"] = _summarize_throughput(
            results=results,
            fields=throughput_cfg.get("fields", []),
            aggregates=throughput_cfg.get("aggregates", []),
            percentiles=throughput_cfg.get("percentiles", []),
            invalid_throughput_samples=invalid_throughput_samples,
            dataset=dataset,
        )

    usage_cfg = profile.get("usage", {})
    if usage_cfg.get("enabled", False):
        metrics["usage"] = _summarize_direct_fields(
            results=results,
            fields=usage_cfg.get("fields", []),
            aggregates=usage_cfg.get("aggregates", []),
            percentiles=usage_cfg.get("percentiles", []),
            missing_fields=missing_fields,
            dataset=dataset,
        )

    telemetry_cfg = profile.get("telemetry", {})
    if telemetry_cfg.get("enabled", False):
        metrics["telemetry"] = _summarize_direct_fields(
            results=results,
            fields=telemetry_cfg.get("fields", []),
            aggregates=telemetry_cfg.get("aggregates", []),
            percentiles=telemetry_cfg.get("percentiles", []),
            missing_fields=missing_fields,
            dataset=dataset,
        )

    return group_summary


def _summarize_direct_fields(
    results: list[BenchmarkResult],
    fields: Iterable[str],
    aggregates: Iterable[str],
    percentiles: Iterable[int],
    missing_fields: list[dict[str, Any]],
    dataset: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for field_path in fields:
        values = []
        for result in results:
            value, found = _get_field_value(result, field_path)
            if found:
                values.append(value)
            else:
                missing_fields.append(
                    {
                        "field": field_path,
                        "sample_id": result.sample_id,
                        "dataset": dataset,
                    }
                )

        summary[field_path] = aggregate_values(values, list(aggregates), list(percentiles))
    return summary


def _summarize_inter_token_latency(
    results: list[BenchmarkResult],
    source_field: str,
    request_level_stats: Iterable[str],
    aggregates: Iterable[str],
    percentiles: Iterable[int],
    missing_fields: list[dict[str, Any]],
    dataset: str,
) -> dict[str, Any]:
    request_stats: dict[str, list[float]] = defaultdict(list)

    for result in results:
        value, found = _get_field_value(result, source_field)
        if not found or not isinstance(value, list):
            missing_fields.append(
                {
                    "field": source_field,
                    "sample_id": result.sample_id,
                    "dataset": dataset,
                }
            )
            continue

        for stat_name in request_level_stats:
            stat_value = _compute_request_stat(value, stat_name)
            if stat_value is not None:
                request_stats[stat_name].append(stat_value)

    summary: dict[str, Any] = {}
    for stat_name in request_level_stats:
        summary[stat_name] = aggregate_values(
            request_stats.get(stat_name, []),
            list(aggregates),
            list(percentiles),
        )

    return summary


def _summarize_throughput(
    results: list[BenchmarkResult],
    fields: Iterable[str],
    aggregates: Iterable[str],
    percentiles: Iterable[int],
    invalid_throughput_samples: set[str],
    dataset: str,
) -> dict[str, Any]:
    derived_values: dict[str, list[float]] = defaultdict(list)

    for result in results:
        e2e = _safe_divide(result.output_tokens, result.total_latency_ms / 1000.0)
        prefill = _safe_divide(result.input_tokens, result.ttft_ms / 1000.0)
        decode_denom = (result.total_latency_ms - result.ttft_ms) / 1000.0
        decoding = _safe_divide(result.output_tokens, decode_denom)

        mapping = {
            "e2e_throughput": e2e,
            "prefill_throughput": prefill,
            "decoding_throughput": decoding,
        }

        for field_name in fields:
            value = mapping.get(field_name)
            if value is None:
                invalid_throughput_samples.add(result.sample_id)
                continue
            derived_values[field_name].append(value)

    summary: dict[str, Any] = {}
    for field_name in fields:
        summary[field_name] = aggregate_values(
            derived_values.get(field_name, []),
            list(aggregates),
            list(percentiles),
        )

    return summary


def _compute_request_stat(values: list[float], stat_name: str) -> float | None:
    if stat_name == "mean":
        return aggregate_values(values, ["mean"])["mean"]
    if stat_name == "p50":
        return percentile(values, 50)
    if stat_name == "p95":
        return percentile(values, 95)
    if stat_name == "p99":
        return percentile(values, 99)
    raise ValueError(f"Unsupported request-level stat: {stat_name}")


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _get_field_value(result: BenchmarkResult, field_path: str) -> tuple[Any, bool]:
    current: Any = result
    for part in field_path.split("."):
        if isinstance(current, BenchmarkResult):
            if not hasattr(current, part):
                return None, False
            current = getattr(current, part)
        elif isinstance(current, Mapping):
            if part not in current:
                return None, False
            current = current[part]
        else:
            return None, False
    return current, True



def _build_quality_summary(
    missing_fields: list[dict[str, Any]],
    invalid_throughput_samples: set[str],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    quality_cfg = profile.get("quality_checks", {})
    return {
        "fail_on_missing_fields": bool(quality_cfg.get("fail_on_missing_fields", False)),
        "missing_field_count": len(missing_fields),
        "invalid_throughput_sample_count": len(invalid_throughput_samples),
        "missing_fields": missing_fields[:200],
    }