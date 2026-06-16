import json
import random
import re
from pathlib import Path

QUESTION_STEMS = [
    "What is the most likely diagnosis?",
    "Which of the following is the most probable diagnosis?",
    "Based on the patient's presentation, what is the most likely diagnosis?",
    "The clinical picture is most consistent with which of the following?",
    "What is the most likely cause of this patient's symptoms?",
    "Which of the following best explains the patient's presentation?",
    "What condition is this patient most likely suffering from?",
    "Which diagnosis best accounts for all of the findings described?",
    "What is the most likely underlying etiology in this case?",
    "This patient's history and findings are most suggestive of which condition?"
]

random.seed(13)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_DIAGNOSIS_LOOKUP_FILES = [
    _PROJECT_ROOT / "data/semi_processed/medcasemcq/eval/formatted_diagnoses_test.jsonl",
    _PROJECT_ROOT / "data/semi_processed/medcasemcq/fewshot/formatted_diagnoses_val.jsonl",
]


def _load_diagnosis_map() -> dict:
    diagnosis_map = {}
    for path in _DIAGNOSIS_LOOKUP_FILES:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                diagnosis_map[entry["pmc_id"]] = entry["formatted_diagnosis"]
    return diagnosis_map


def _normalize_casing_leaks(text: str) -> str:
    """Forces 'Syndrome' to lowercase unless it is the first word."""
    return re.sub(r'(?<=\w\s)Syndrome\b', 'syndrome', text)


_DIAGNOSIS_MAP = _load_diagnosis_map()


def preprocess(row: dict) -> dict:
    pmcid = row.get("pmcid", "")

    if pmcid in _DIAGNOSIS_MAP:
        row["final_diagnosis"] = _normalize_casing_leaks(_DIAGNOSIS_MAP[pmcid])

    stem = random.choice(QUESTION_STEMS)
    row["case_prompt"] = f"{row['case_prompt']}\n\n{stem}"

    return row
