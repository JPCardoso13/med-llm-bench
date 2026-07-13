from __future__ import annotations

import json
import re
from typing import Any, Mapping

from llm_bench.judge.schemas import validate_judge_response

_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_judge_response(raw_text: str, rubric: Mapping[str, Any]) -> dict | None:
    cleaned = _CODE_FENCE_PATTERN.sub("", raw_text.strip()).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    return validate_judge_response(payload, rubric)
