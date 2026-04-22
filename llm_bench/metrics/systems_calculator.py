from __future__ import annotations

from collections import defaultdict
from itertools import product
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

    group_fields = list(profile.get("group_by", []))
    groups: dict[tuple[Any, ...], list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        for group_key in _build_group_keys(result, group_fields):
            groups[group_key].append(result)

    missing_fields: list[dict[str, Any]] = []
    invalid_throughput_samples: set[str] = set()
    group_summaries = []

    for group_key, group_results in groups.items():
        group_summaries.append(
            _summarize_group(
                group_key=group_key,
                results=group_results,
                profile=profile,
                missing_fields=missing_fields,
                invalid_throughput_samples=invalid_throughput_samples,
            )
        )

    return {
        "profile_id": profile.get("profile_id", "systems"),
        "schema_version": profile.get("schema_version", 1),
        "scope": profile.get("scope", "systems"),
        "enabled": True,
        "sample_count": len(results),
        "group_by": group_fields,
        "groups": group_summaries,
        "quality": _build_quality_summary(
            missing_fields=missing_fields,
            invalid_throughput_samples=invalid_throughput_samples,
            profile=profile,
        ),
    }


def _summarize_group(
    group_key: tuple[Any, ...],
    results: list[BenchmarkResult],
    profile: Mapping[str, Any],
    missing_fields: list[dict[str, Any]],
    invalid_throughput_samples: set[str],
) -> dict[str, Any]:
    group_label = _group_key_to_label(group_key, profile.get("group_by", []))
    group_summary: dict[str, Any] = {
        "group_key": group_label,
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
            group_key=group_key,
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
            group_key=group_key,
        )

    throughput_cfg = profile.get("throughput", {})
    if throughput_cfg.get("enabled", False):
        metrics["throughput"] = _summarize_throughput(
            results=results,
            fields=throughput_cfg.get("fields", []),
            aggregates=throughput_cfg.get("aggregates", []),
            percentiles=throughput_cfg.get("percentiles", []),
            invalid_throughput_samples=invalid_throughput_samples,
            group_key=group_key,
        )

    usage_cfg = profile.get("usage", {})
    if usage_cfg.get("enabled", False):
        metrics["usage"] = _summarize_direct_fields(
            results=results,
            fields=usage_cfg.get("fields", []),
            aggregates=usage_cfg.get("aggregates", []),
            percentiles=usage_cfg.get("percentiles", []),
            missing_fields=missing_fields,
            group_key=group_key,
        )

    telemetry_cfg = profile.get("telemetry", {})
    if telemetry_cfg.get("enabled", False):
        metrics["telemetry"] = _summarize_direct_fields(
            results=results,
            fields=telemetry_cfg.get("fields", []),
            aggregates=telemetry_cfg.get("aggregates", []),
            percentiles=telemetry_cfg.get("percentiles", []),
            missing_fields=missing_fields,
            group_key=group_key,
        )

    return group_summary


def _summarize_direct_fields(
    results: list[BenchmarkResult],
    fields: Iterable[str],
    aggregates: Iterable[str],
    percentiles: Iterable[int],
    missing_fields: list[dict[str, Any]],
    group_key: tuple[Any, ...],
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
                        "group_key": [str(part) for part in group_key],
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
    group_key: tuple[Any, ...],
) -> dict[str, Any]:
    request_stats: dict[str, list[float]] = defaultdict(list)

    for result in results:
        value, found = _get_field_value(result, source_field)
        if not found or not isinstance(value, list):
            missing_fields.append(
                {
                    "field": source_field,
                    "sample_id": result.sample_id,
                    "group_key": [str(part) for part in group_key],
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
    group_key: tuple[Any, ...],
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


def _build_group_keys(result: BenchmarkResult, group_fields: Iterable[str]) -> list[tuple[Any, ...]]:
    fields = list(group_fields)
    if not fields:
        return [tuple()]

    per_field_values: list[list[Any]] = []
    for field in fields:
        values = _resolve_group_field_values(result, field)
        per_field_values.append(values if values else [None])

    return [tuple(parts) for parts in product(*per_field_values)]


def _resolve_group_field_values(result: BenchmarkResult, field: str) -> list[Any]:
    value, found = _get_field_value(result, field)
    if not found:
        value, found = _get_field_value(result, f"grouping.{field}")

    if not found:
        return [None]

    if isinstance(value, list):
        if not value:
            return [None]
        # Keep insertion order but avoid duplicate labels in one sample.
        return list(dict.fromkeys(value))

    return [value]


def _group_key_to_label(group_key: tuple[Any, ...], group_fields: Iterable[str]) -> dict[str, Any]:
    return {field: value for field, value in zip(group_fields, group_key, strict=False)}


def _build_quality_summary(
    missing_fields: list[dict[str, Any]],
    invalid_throughput_samples: set[str],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    quality_cfg = profile.get("quality_checks", {})
    return {
        "fail_on_missing_fields": bool(quality_cfg.get("fail_on_missing_fields", False)),
        "require_system_telemetry_ok": bool(quality_cfg.get("require_system_telemetry_ok", False)),
        "warn_on_empty_input": bool(quality_cfg.get("warn_on_empty_input", True)),
        "warn_on_missing_telemetry": bool(quality_cfg.get("warn_on_missing_telemetry", True)),
        "missing_field_count": len(missing_fields),
        "invalid_throughput_sample_count": len(invalid_throughput_samples),
        "missing_fields": missing_fields[:200],
    }