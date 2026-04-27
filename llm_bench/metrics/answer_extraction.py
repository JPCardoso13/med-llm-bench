from __future__ import annotations

import re
from typing import Any


_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAIL_PATTERN = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)

# Ordered patterns from most explicit to most permissive.
_ANSWER_PATTERNS = [
    re.compile(r"final\s+answer\s*[:\-]?\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE),
    re.compile(r"answer\s*[:\-]?\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE),
    re.compile(r"option\s*[:\-]?\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?\s*([A-J])\s*\)?\s*$", re.IGNORECASE | re.MULTILINE),
]


def clean_response_text(text: str) -> str:
    """Remove think blocks and normalize whitespace for extraction."""
    cleaned = _THINK_BLOCK_PATTERN.sub(" ", text)
    cleaned = _THINK_TAIL_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def extract_mcq_answer_letter(text: str) -> dict[str, Any]:
    """Extract MCQ answer letter from model output.

    Returns a dictionary with:
    - letter: extracted answer letter or None
    - status: one of success | ambiguous | missing
    - candidates: ordered unique candidate letters found
    - cleaned_text: response text after removing think blocks
    """
    cleaned = clean_response_text(text)

    candidates: list[str] = []
    seen: set[str] = set()

    for pattern in _ANSWER_PATTERNS:
        for match in pattern.findall(cleaned):
            letter = str(match).upper()
            if letter not in seen:
                seen.add(letter)
                candidates.append(letter)

    if not candidates:
        return {
            "letter": None,
            "status": "missing",
            "candidates": [],
            "cleaned_text": cleaned,
        }

    if len(candidates) == 1:
        return {
            "letter": candidates[0],
            "status": "success",
            "candidates": candidates,
            "cleaned_text": cleaned,
        }

    return {
        "letter": None,
        "status": "ambiguous",
        "candidates": candidates,
        "cleaned_text": cleaned,
    }
