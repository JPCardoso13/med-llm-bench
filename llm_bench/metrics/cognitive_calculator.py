from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import re
import string
from typing import Any, Iterable, Mapping

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
            parse_failures=parse_failures,
            ambiguous_extractions=ambiguous_extractions,
            missing_ref_fields=missing_ref_fields,
            dataset=dataset,
        )
    elif task_type == "generative":
        group_summary["metrics"]["generative"] = _summarize_generative_group(
            results=results,
            profile=profile,
            parse_failures=parse_failures,
            missing_ref_fields=missing_ref_fields,
            dataset=dataset,
        )

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
    loose_match_cfg = generative_cfg.get("loose_match", {})
    semantic_similarity_cfg = profile.get("semantic_similarity", {})

    answer_scores: list[float] = []
    explanation_loose_scores: list[float] = []
    rouge_scores: list[float] = []
    bertscore_scores: list[float] = []
    bertscore_threshold = semantic_similarity_cfg.get("bertscore", {}).get("threshold")

    evaluated_count = 0
    parsed_count = 0

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
            continue

        if not ref_reasoning:
            missing_ref_fields.append(
                {
                    "sample_id": result.sample_id,
                    "dataset": dataset,
                    "field": "ref_fields.ref_reasoning",
                }
            )
            continue

        extracted = _extract_generative_response(result.response, profile)
        predicted_answer = extracted["final_answer"]
        predicted_explanation = extracted["explanation"]

        if predicted_answer is None or predicted_explanation is None:
            parse_failures.append(
                {
                    "sample_id": result.sample_id,
                    "dataset": dataset,
                    "status": "missing",
                    "missing_fields": [
                        name
                        for name, value in (("final_answer", predicted_answer), ("explanation", predicted_explanation))
                        if value is None
                    ],
                }
            )
            continue

        parsed_count += 1
        evaluated_count += 1

        answer_scores.append(
            1.0 if _normalize_text(predicted_answer, exact_match_cfg) == _normalize_text(ref_answer, exact_match_cfg) else 0.0
        )

        explanation_loose_scores.append(
            _sequence_similarity(
                _normalize_text(predicted_explanation, loose_match_cfg),
                _normalize_text(ref_reasoning, loose_match_cfg),
            )
        )

        rouge_scores.append(
            _token_f1_score(
                _normalize_text(predicted_explanation, semantic_similarity_cfg.get("rouge", {})),
                _normalize_text(ref_reasoning, semantic_similarity_cfg.get("rouge", {})),
            )
        )

        bertscore_scores.append(
            _sequence_similarity(
                _normalize_text(predicted_explanation, semantic_similarity_cfg.get("bertscore", {})),
                _normalize_text(ref_reasoning, semantic_similarity_cfg.get("bertscore", {})),
            )
        )

    answer_summary = aggregate_values(answer_scores, ["mean", "min", "max"])
    explanation_loose_summary = aggregate_values(explanation_loose_scores, ["mean", "min", "max"])
    rouge_summary = aggregate_values(rouge_scores, ["mean", "min", "max"])
    bertscore_summary = aggregate_values(bertscore_scores, ["mean", "min", "max"])

    answer_summary["correct_count"] = int(sum(answer_scores))
    answer_summary["evaluated_count"] = len(answer_scores)
    answer_summary["accuracy"] = (
        answer_summary["correct_count"] / answer_summary["evaluated_count"] if answer_summary["evaluated_count"] > 0 else None
    )

    explanation_loose_summary["evaluated_count"] = len(explanation_loose_scores)
    rouge_summary["evaluated_count"] = len(rouge_scores)
    bertscore_summary["evaluated_count"] = len(bertscore_scores)

    if bertscore_threshold is not None:
        bertscore_summary["threshold"] = float(bertscore_threshold)
        bertscore_summary["above_threshold_count"] = sum(1 for score in bertscore_scores if score >= float(bertscore_threshold))
        bertscore_summary["above_threshold_rate"] = (
            bertscore_summary["above_threshold_count"] / bertscore_summary["evaluated_count"]
            if bertscore_summary["evaluated_count"] > 0
            else None
        )

    return {
        "dataset": dataset,
        "sample_count": len(results),
        "metrics": {
            "answer": {
                "exact_match": answer_summary,
            },
            "explanation": {
                "loose_match": explanation_loose_summary,
                "semantic_similarity": {
                    "rouge": rouge_summary,
                    "bertscore": bertscore_summary,
                },
            },
        },
        "quality": {
            "parse_failure_count": len(parse_failures),
            "missing_ref_field_count": len(missing_ref_fields),
            "parsed_count": parsed_count,
        },
    }


def _extract_generative_response(response: str, profile: Mapping[str, Any]) -> dict[str, str | None]:
    cleaned = clean_response_text(response)
    extraction_cfg = profile.get("extraction", {})
    final_answer = _extract_section(cleaned, extraction_cfg.get("final_answer", {}), fallback_label="final answer")
    explanation = _extract_section(cleaned, extraction_cfg.get("explanation", {}), fallback_label="explanation")
    return {
        "final_answer": final_answer,
        "explanation": explanation,
    }


def _extract_section(text: str, section_cfg: Mapping[str, Any], fallback_label: str) -> str | None:
    value = _extract_by_regex(text, section_cfg.get("pattern"))
    if value is None:
        value = _extract_by_regex(text, rf"(?is){re.escape(fallback_label)}\s*[:\-\s]*(.*)$")
    if value is None:
        return None
    if section_cfg.get("trim", True):
        value = value.strip()
    return value or None


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


def _sequence_similarity(prediction: str, reference: str) -> float:
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
    return SequenceMatcher(None, prediction, reference).ratio()


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
