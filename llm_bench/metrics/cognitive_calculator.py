from __future__ import annotations

from collections import defaultdict
import re
import string
from typing import Any, Mapping

from llm_bench.metrics.answer_extraction import extract_mcq_answer_letter
from llm_bench.metrics.answer_extraction import clean_response_text
from llm_bench.metrics.stats import aggregate_values
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

    groups: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        groups[result.dataset].append(result)

    parse_failures: list[dict[str, Any]] = []
    ambiguous_extractions: list[dict[str, Any]] = []
    missing_ref_fields: list[dict[str, Any]] = []

    group_summaries = []
    for dataset, group_results in groups.items():
        group_summaries.append(
            _summarize_group(
                dataset=dataset,
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
        "groups": group_summaries,
        "quality": quality,
    }


def _summarize_group(
    dataset: str,
    results: list[BenchmarkResult],
    profile: Mapping[str, Any],
    parse_failures: list[dict[str, Any]],
    ambiguous_extractions: list[dict[str, Any]],
    missing_ref_fields: list[dict[str, Any]],
) -> dict[str, Any]:
    group_parse_failures: list[dict[str, Any]] = []
    group_ambiguous_extractions: list[dict[str, Any]] = []
    group_missing_ref_fields: list[dict[str, Any]] = []

    group_summary: dict[str, Any] = {
        "dataset": dataset,
        "sample_count": len(results),
        "metrics": {},
    }

    task_type = str(profile.get("task_type", "mcq")).lower()

    if task_type == "mcq":
        group_summary["metrics"]["mcq"] = _summarize_mcq_group(
            results=results,
            profile=profile,
            parse_failures=group_parse_failures,
            ambiguous_extractions=group_ambiguous_extractions,
            missing_ref_fields=group_missing_ref_fields,
            dataset=dataset,
        )
    elif task_type == "generative":
        group_summary["metrics"]["generative"] = _summarize_generative_group(
            results=results,
            profile=profile,
            parse_failures=group_parse_failures,
            missing_ref_fields=group_missing_ref_fields,
            dataset=dataset,
        )

    parse_failures.extend(group_parse_failures)
    ambiguous_extractions.extend(group_ambiguous_extractions)
    missing_ref_fields.extend(group_missing_ref_fields)

    group_summary["quality"] = {
        "parse_failure_count": len(group_parse_failures),
        "ambiguous_extraction_count": len(group_ambiguous_extractions),
        "missing_ref_field_count": len(group_missing_ref_fields),
    }

    return group_summary


def _summarize_generative_group(
    results: list[BenchmarkResult],
    profile: Mapping[str, Any],
    parse_failures: list[dict[str, Any]],
    missing_ref_fields: list[dict[str, Any]],
    dataset: str,
) -> dict[str, Any]:
    generative_cfg = profile.get("generative", {})
    if not generative_cfg.get("enabled", True):
        return {}

    exact_match_cfg = generative_cfg.get("exact_match", {})
    answer_token_f1_cfg = generative_cfg.get("answer_token_f1", {})
    reporting_cfg = profile.get("reporting", {})
    expl_token_f1_cfg = profile.get("similarity", {}).get("token_f1", {})

    answer_exact_scores: list[float] = []
    answer_token_f1_scores: list[float] = []
    explanation_token_f1_scores: list[float] = []

    evaluated_count = 0
    parsed_count = 0
    contaminated_count = 0
    per_sample_rows: list[dict[str, Any]] = []

    for result in results:
        ref_answer = str(result.ref_fields.get("answer", "")).strip()
        ref_reasoning = str(result.ref_fields.get("ref_reasoning", "")).strip()

        if not ref_answer:
            missing_ref_fields.append(
                {
                    "sample_id": result.sample_id,
                    "dataset": dataset,
                    "field": "ref_fields.answer",
                }
            )
            per_sample_rows.append(
                {
                    "sample_id": result.sample_id,
                    "dataset": dataset,
                    "status": "missing_ref_fields",
                    "missing_fields": ["ref_fields.answer"],
                }
            )
            continue

        extracted = _extract_generative_response(result.response, profile)
        predicted_answer = extracted["final_answer"]
        predicted_explanation = extracted["explanation"]
        answer_contaminated = bool(extracted.get("answer_contaminated", False))

        if predicted_answer is None:
            parse_failures.append(
                {
                    "sample_id": result.sample_id,
                    "dataset": dataset,
                    "status": "missing",
                    "missing_fields": ["final_answer"],
                }
            )
            per_sample_rows.append(
                {
                    "sample_id": result.sample_id,
                    "dataset": dataset,
                    "status": "parse_failure",
                    "missing_fields": ["final_answer"],
                }
            )
            continue

        parsed_count += 1
        evaluated_count += 1
        if answer_contaminated:
            contaminated_count += 1

        answer_exact = (
            1.0 if _normalize_text(predicted_answer, exact_match_cfg) == _normalize_text(ref_answer, exact_match_cfg) else 0.0
        )
        answer_f1 = _token_f1_score(
            _normalize_text(predicted_answer, answer_token_f1_cfg),
            _normalize_text(ref_answer, answer_token_f1_cfg),
        )

        answer_exact_scores.append(answer_exact)
        answer_token_f1_scores.append(answer_f1)

        explanation_token_f1 = None
        if ref_reasoning and predicted_explanation:
            explanation_token_f1 = _token_f1_score(
                _normalize_text(predicted_explanation, expl_token_f1_cfg),
                _normalize_text(ref_reasoning, expl_token_f1_cfg),
            )
            explanation_token_f1_scores.append(explanation_token_f1)

        composite_score = 0.5 * answer_exact + 0.5 * answer_f1
        per_sample_rows.append(
            {
                "sample_id": result.sample_id,
                "dataset": dataset,
                "status": "ok",
                "answer_exact": answer_exact,
                "answer_token_f1": answer_f1,
                "format_ok": not answer_contaminated,
                "answer_contaminated": answer_contaminated,
                "explanation_token_f1": explanation_token_f1,
                "composite_score": composite_score,
                "response_preview": _safe_preview(result.response, reporting_cfg),
                "final_answer": _safe_preview(predicted_answer, reporting_cfg),
                "reference_answer": _safe_preview(ref_answer, reporting_cfg),
                "explanation_preview": _safe_preview(predicted_explanation, reporting_cfg),
            }
        )

    answer_exact_summary = aggregate_values(answer_exact_scores, ["mean", "min", "max"])
    answer_token_f1_summary = aggregate_values(answer_token_f1_scores, ["mean", "min", "max"])
    explanation_token_f1_summary = aggregate_values(explanation_token_f1_scores, ["mean", "min", "max"])

    answer_exact_summary["correct_count"] = int(sum(answer_exact_scores))
    answer_exact_summary["evaluated_count"] = len(answer_exact_scores)
    answer_exact_summary["accuracy"] = (
        answer_exact_summary["correct_count"] / answer_exact_summary["evaluated_count"]
        if answer_exact_summary["evaluated_count"] > 0
        else None
    )

    answer_token_f1_summary["evaluated_count"] = len(answer_token_f1_scores)
    explanation_token_f1_summary["evaluated_count"] = len(explanation_token_f1_scores)

    format_summary = {
        "parse_success_count": parsed_count,
        "parse_failure_count": len(parse_failures),
        "answer_contaminated_count": contaminated_count,
        "parse_success_rate": (parsed_count / len(results)) if results else None,
        "answer_clean_rate": (1.0 - (contaminated_count / parsed_count)) if parsed_count > 0 else None,
    }

    summary = {
        "dataset": dataset,
        "sample_count": len(results),
        "metrics": {
            "answer": {
                "exact_match": answer_exact_summary,
                "token_f1": answer_token_f1_summary,
            },
            "format": format_summary,
            "explanation": {
                "token_f1": explanation_token_f1_summary,
            },
        },
    }

    diagnostics = _build_generative_diagnostics(per_sample_rows, reporting_cfg)
    if diagnostics:
        summary["diagnostics"] = diagnostics

    return summary


def _build_generative_diagnostics(rows: list[dict[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not cfg.get("enabled", False):
        return {}

    top_k_worst = int(cfg.get("top_k_worst", 20))
    top_k_best = int(cfg.get("top_k_best", 5))
    include_per_sample = bool(cfg.get("include_per_sample", False))
    per_sample_limit = int(cfg.get("per_sample_limit", 1000))

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    sorted_ok = sorted(ok_rows, key=lambda row: float(row.get("composite_score", 0.0)))

    diagnostics: dict[str, Any] = {
        "configured": {
            "top_k_worst": top_k_worst,
            "top_k_best": top_k_best,
            "include_per_sample": include_per_sample,
            "per_sample_limit": per_sample_limit,
        },
        "worst_samples": sorted_ok[: max(0, top_k_worst)],
        "best_samples": list(reversed(sorted_ok[-max(0, top_k_best):])) if top_k_best > 0 else [],
    }

    if include_per_sample:
        diagnostics["per_sample"] = rows[: max(0, per_sample_limit)]

    return diagnostics


def _safe_preview(value: str | None, cfg: Mapping[str, Any]) -> str:
    if not cfg.get("include_text_preview", True):
        return ""

    text = str(value or "")
    max_chars = int(cfg.get("text_preview_chars", 220))
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _extract_generative_response(response: str, profile: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = clean_response_text(response)
    extraction_cfg = profile.get("extraction", {})
    final_answer = _extract_section(cleaned, extraction_cfg.get("final_answer", {}), fallback_label="final answer")
    explanation = _extract_section(cleaned, extraction_cfg.get("explanation", {}), fallback_label="explanation")
    final_answer, answer_contaminated = _normalize_final_answer(final_answer)
    return {
        "final_answer": final_answer,
        "explanation": explanation,
        "answer_contaminated": answer_contaminated,
    }


def _extract_section(text: str, section_cfg: Mapping[str, Any], fallback_label: str) -> str | None:
    value = _extract_by_regex(text, section_cfg.get("pattern"))
    if value is None:
        if fallback_label == "final answer":
            value = _extract_by_regex(
                text,
                rf"(?is){re.escape(fallback_label)}\s*[:\-\s]*(.*?)(?=\n\s*explanation\s*[:\-]|\bexplanation\s*[:\-]|$)",
            )
            if value is None:
                value = _extract_by_regex(text, rf"(?is){re.escape(fallback_label)}\s*[:\-\s]*(.*)$")
        else:
            value = _extract_by_regex(text, rf"(?is){re.escape(fallback_label)}\s*[:\-\s]*(.*)$")
    if value is None:
        return None
    if section_cfg.get("trim", True):
        value = value.strip()
    return value or None


def _normalize_final_answer(final_answer: str | None) -> tuple[str | None, bool]:
    if final_answer is None:
        return None, False

    value = str(final_answer).strip()
    contaminated = False

    explanation_marker = re.search(r"(?is)\bexplanation\s*[:\-]", value)
    if explanation_marker:
        contaminated = True
        value = value[: explanation_marker.start()].strip()

    if "\n" in value:
        contaminated = True
        value = value.splitlines()[0].strip()

    value = value.strip().rstrip(". ")
    return (value or None), contaminated


def _extract_by_regex(text: str, pattern: str | None) -> str | None:
    if not pattern:
        return None
    match = re.search(pattern, text)
    if not match:
        return None
    if match.groups():
        candidate = match.group(1)
    else:
        candidate = match.group(0)
    return candidate


def _normalize_text(text: str | None, cfg: Mapping[str, Any]) -> str:
    value = str(text or "")
    if cfg.get("lowercase", False):
        value = value.lower()
    if cfg.get("strip_punctuation", False):
        value = value.translate(str.maketrans({char: " " for char in string.punctuation}))
    if cfg.get("normalize_whitespace", True):
        value = re.sub(r"\s+", " ", value).strip()
    return value


def _token_f1_score(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1

    ref_counts: dict[str, int] = defaultdict(int)
    for token in ref_tokens:
        ref_counts[token] += 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _summarize_mcq_group(
    results: list[BenchmarkResult],
    profile: Mapping[str, Any],
    parse_failures: list[dict[str, Any]],
    ambiguous_extractions: list[dict[str, Any]],
    missing_ref_fields: list[dict[str, Any]],
    dataset: str,
) -> dict[str, Any]:
    mcq_cfg = profile.get("mcq", {})
    if not mcq_cfg.get("enabled", True):
        return {}

    enable_accuracy = bool(mcq_cfg.get("accuracy", {}).get("enabled", True))
    enable_precision = bool(mcq_cfg.get("precision", {}).get("enabled", True))
    enable_recall = bool(mcq_cfg.get("recall", {}).get("enabled", True))
    enable_f1 = bool(mcq_cfg.get("f1", {}).get("enabled", True))
    enable_parsing = bool(mcq_cfg.get("parsing", {}).get("enabled", True))

    evaluated_count = 0
    correct_count = 0
    parsed_success_count = 0
    parse_failure_count = 0
    ambiguous_count = 0
    label_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for result in results:
        ref_answer = str(result.ref_fields.get("answer_idx", "")).strip().upper()
        if not ref_answer:
            missing_ref_fields.append(
                {
                    "sample_id": result.sample_id,
                    "dataset": dataset,
                    "field": "ref_fields.answer_idx",
                }
            )
            continue

        extraction = extract_mcq_answer_letter(result.response)
        status = extraction["status"]
        pred = extraction["letter"]

        label_stats.setdefault(ref_answer, {"tp": 0, "fp": 0, "fn": 0})

        if status == "success":
            parsed_success_count += 1
        elif status == "ambiguous":
            ambiguous_count += 1
            ambiguous_extractions.append(
                {
                    "sample_id": result.sample_id,
                    "dataset": dataset,
                    "candidates": extraction.get("candidates", []),
                }
            )
        else:
            parse_failure_count += 1
            parse_failures.append(
                {
                    "sample_id": result.sample_id,
                    "dataset": dataset,
                    "status": status,
                }
            )

        evaluated_count += 1
        if pred is not None:
            label_stats.setdefault(pred, {"tp": 0, "fp": 0, "fn": 0})

        if pred is not None and pred == ref_answer:
            correct_count += 1
            label_stats[ref_answer]["tp"] += 1
        else:
            label_stats[ref_answer]["fn"] += 1
            if pred is not None:
                label_stats[pred]["fp"] += 1

    summary: dict[str, Any] = {}

    if enable_accuracy:
        summary["accuracy"] = {
            "evaluated_count": evaluated_count,
            "correct_count": correct_count,
            "accuracy": (correct_count / evaluated_count) if evaluated_count > 0 else None,
        }

    if enable_precision or enable_recall or enable_f1:
        classification_summary = _build_mcq_classification_summary(label_stats)

        if enable_precision:
            summary["precision"] = classification_summary["precision"]

        if enable_recall:
            summary["recall"] = classification_summary["recall"]

        if enable_f1:
            summary["f1"] = classification_summary["f1"]

    if enable_parsing:
        summary["parsing"] = {
            "parsed_success_count": parsed_success_count,
            "parse_failure_count": parse_failure_count,
            "ambiguous_count": ambiguous_count,
            "parse_success_rate": (parsed_success_count / evaluated_count) if evaluated_count > 0 else None,
        }

    return summary


def _build_mcq_classification_summary(label_stats: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    labels = sorted(label_stats.keys())

    per_label: dict[str, dict[str, Any]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    for label in labels:
        stats = label_stats[label]
        tp = int(stats.get("tp", 0))
        fp = int(stats.get("fp", 0))
        fn = int(stats.get("fn", 0))

        precision_denominator = tp + fp
        recall_denominator = tp + fn
        precision = (tp / precision_denominator) if precision_denominator > 0 else 0.0
        recall = (tp / recall_denominator) if recall_denominator > 0 else 0.0
        f1_denominator = precision + recall
        f1 = (2.0 * precision * recall / f1_denominator) if f1_denominator > 0 else 0.0

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "labels": labels,
        "average": "macro",
        "precision": {
            "value": (sum(precision_values) / len(precision_values)) if precision_values else None,
            "per_label": per_label,
        },
        "recall": {
            "value": (sum(recall_values) / len(recall_values)) if recall_values else None,
            "per_label": per_label,
        },
        "f1": {
            "value": (sum(f1_values) / len(f1_values)) if f1_values else None,
            "per_label": per_label,
        },
    }


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
