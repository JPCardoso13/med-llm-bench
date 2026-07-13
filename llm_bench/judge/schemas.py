from __future__ import annotations

from typing import Any, Mapping


def build_judge_response_format(task_id: str, rubric: Mapping[str, Any]) -> dict:
    """Build a vLLM response_format (json_schema) from a profile's llm_judge.rubric block."""
    properties = {
        item_name: {"type": "string", "enum": list(item_cfg["labels"])}
        for item_name, item_cfg in rubric.items()
        if item_cfg.get("enabled", True)
    }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{task_id}_judge_score",
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def validate_judge_response(payload: Mapping[str, Any], rubric: Mapping[str, Any]) -> dict | None:
    """Validate a parsed judge JSON payload against a profile's llm_judge.rubric block."""
    result: dict[str, Any] = {}
    for item_name, item_cfg in rubric.items():
        if not item_cfg.get("enabled", True):
            continue
        value = payload.get(item_name)
        if value not in item_cfg["labels"]:
            return None
        result[item_name] = value
    return result
