from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any, Iterable, Mapping

from llm_bench.metrics.answer_extraction import extract_mcq_answer_letter
from llm_bench.schemas.benchmark_result import BenchmarkResult


def calculate_cognitive_metrics(results: list[BenchmarkResult], profile: Mapping[str, Any]) -> dict[str, Any]:
    if not profile.get("enabled", True):
        return {
            "profile_id": profile.get("profile_id", "cognitive"),
            "schema_version": profile.get("schema_version", 1),
            "scope": profile.get("scope", "cognitive"),
            "enabled": False,
            "sample_count": len(results),
            "groups": [],
        }

    group_fields = list(profile.get("group_by", ["task_name", "dataset", "model_id", "backend"]))

    groups: dict[tuple[Any, ...], list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        for group_key in _build_group_keys(result, group_fields):
            groups[group_key].append(result)

    parse_failures: list[dict[str, Any]] = []
    ambiguous_extractions: list[dict[str, Any]] = []
    missing_ref_fields: list[dict[str, Any]] = []

    group_summaries = []
    for group_key, group_results in groups.items():
        group_summaries.append(
            _summarize_group(
                group_key=group_key,
                results=group_results,
                profile=profile,
                parse_failures=parse_failures,
                ambiguous_extractions=ambiguous_extractions,
                missing_ref_fields=missing_ref_fields,
            )
        )

    quality = _build_quality_summary(
        profile=profile,
        parse_failures=parse_failures,
        ambiguous_extractions=ambiguous_extractions,
        missing_ref_fields=missing_ref_fields,
    )

    fail_on_missing_ref_fields = bool(profile.get("quality_checks", {}).get("fail_on_missing_ref_fields", False))
    if fail_on_missing_ref_fields and quality["missing_ref_field_count"] > 0:
        raise ValueError(
            "Cognitive metrics failed: missing reference fields detected "
            f"({quality['missing_ref_field_count']} samples)."
        )

    return {
        "profile_id": profile.get("profile_id", "cognitive"),
        "schema_version": profile.get("schema_version", 1),
        "scope": profile.get("scope", "cognitive"),
        "task_type": profile.get("task_type", "mcq"),
        "enabled": True,
        "sample_count": len(results),
        "group_by": group_fields,
        "groups": group_summaries,
        "quality": quality,
    }


def _summarize_group(
    group_key: tuple[Any, ...],
    results: list[BenchmarkResult],
    profile: Mapping[str, Any],
    parse_failures: list[dict[str, Any]],
    ambiguous_extractions: list[dict[str, Any]],
    missing_ref_fields: list[dict[str, Any]],
) -> dict[str, Any]:
    group_summary: dict[str, Any] = {
        "group_key": _group_key_to_label(group_key, profile.get("group_by", [])),
        "sample_count": len(results),
        "metrics": {},
    }

    task_type = str(profile.get("task_type", "mcq")).lower()

    if task_type == "mcq":
        group_summary["metrics"]["mcq"] = _summarize_mcq_group(
            results=results,
            profile=profile,
            parse_failures=parse_failures,
            ambiguous_extractions=ambiguous_extractions,
            missing_ref_fields=missing_ref_fields,
            group_key=group_key,
        )

    return group_summary


def _summarize_mcq_group(
    results: list[BenchmarkResult],
    profile: Mapping[str, Any],
    parse_failures: list[dict[str, Any]],
    ambiguous_extractions: list[dict[str, Any]],
    missing_ref_fields: list[dict[str, Any]],
    group_key: tuple[Any, ...],
) -> dict[str, Any]:
    mcq_cfg = profile.get("mcq", {})
    if not mcq_cfg.get("enabled", True):
        return {}

    enable_accuracy = bool(mcq_cfg.get("accuracy", {}).get("enabled", True))
    enable_parsing = bool(mcq_cfg.get("parsing", {}).get("enabled", True))

    evaluated_count = 0
    correct_count = 0
    parsed_success_count = 0
    parse_failure_count = 0
    ambiguous_count = 0

    for result in results:
        ref_answer = str(result.ref_fields.get("answer_idx", "")).strip().upper()
        if not ref_answer:
            missing_ref_fields.append(
                {
                    "sample_id": result.sample_id,
                    "group_key": [str(part) for part in group_key],
                    "field": "ref_fields.answer_idx",
                }
            )
            continue

        extraction = extract_mcq_answer_letter(result.response)
        status = extraction["status"]
        pred = extraction["letter"]

        if status == "success":
            parsed_success_count += 1
        elif status == "ambiguous":
            ambiguous_count += 1
            ambiguous_extractions.append(
                {
                    "sample_id": result.sample_id,
                    "group_key": [str(part) for part in group_key],
                    "candidates": extraction.get("candidates", []),
                }
            )
        else:
            parse_failure_count += 1
            parse_failures.append(
                {
                    "sample_id": result.sample_id,
                    "group_key": [str(part) for part in group_key],
                    "status": status,
                }
            )

        evaluated_count += 1
        if pred is not None and pred == ref_answer:
            correct_count += 1

    summary: dict[str, Any] = {}

    if enable_accuracy:
        summary["accuracy"] = {
            "evaluated_count": evaluated_count,
            "correct_count": correct_count,
            "accuracy": (correct_count / evaluated_count) if evaluated_count > 0 else None,
        }

    if enable_parsing:
        summary["parsing"] = {
            "parsed_success_count": parsed_success_count,
            "parse_failure_count": parse_failure_count,
            "ambiguous_count": ambiguous_count,
            "parse_success_rate": (parsed_success_count / evaluated_count) if evaluated_count > 0 else None,
        }

    return summary


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
        return list(dict.fromkeys(value))

    return [value]


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


def _group_key_to_label(group_key: tuple[Any, ...], group_fields: Iterable[str]) -> dict[str, Any]:
    return {field: value for field, value in zip(group_fields, group_key, strict=False)}


def _build_quality_summary(
    profile: Mapping[str, Any],
    parse_failures: list[dict[str, Any]],
    ambiguous_extractions: list[dict[str, Any]],
    missing_ref_fields: list[dict[str, Any]],
) -> dict[str, Any]:
    quality_cfg = profile.get("quality_checks", {})
    return {
        "fail_on_missing_ref_fields": bool(quality_cfg.get("fail_on_missing_ref_fields", False)),
        "warn_on_parse_failures": bool(quality_cfg.get("warn_on_parse_failures", True)),
        "warn_on_extraction_ambiguity": bool(quality_cfg.get("warn_on_extraction_ambiguity", True)),
        "parse_failure_count": len(parse_failures),
        "ambiguous_extraction_count": len(ambiguous_extractions),
        "missing_ref_field_count": len(missing_ref_fields),
        "parse_failures": parse_failures[:200],
        "ambiguous_extractions": ambiguous_extractions[:200],
        "missing_ref_fields": missing_ref_fields[:200],
    }
