import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
DEFAULT_OUTPUT_PATH = "data/interim/medcasereasoning/mcq_dataset.jsonl"
MAX_RETRIES = 3
TEMPERATURE_SCHEDULE = [0.7, 0.5, 0.3]

SYSTEM_PROMPT = """You are an expert clinical reasoning assistant.

Task:
Given a patient case prompt, diagnostic reasoning, and the final diagnosis, produce exactly 3 incorrect but clinically plausible differential diagnoses (distractors).

Rules:
1. Prefer extracting distractors that are explicitly mentioned in diagnostic_reasoning.
2. If fewer than 3 are available, generate plausible distractors from case_prompt.
3. Distractors must NOT be the same as, or aliases/synonyms of final_diagnosis.
4. Return ONLY valid JSON in this exact format:
{"distractors": ["d1", "d2", "d3"]}
"""


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def extract_json(text: str) -> Optional[Dict[str, object]]:
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def validate_distractors(final_diagnosis: str, distractors: object) -> Tuple[bool, str]:
    if not isinstance(distractors, list):
        return False, "Missing or invalid 'distractors' list."

    if len(distractors) != 3:
        return False, f"Expected exactly 3 distractors, got {len(distractors)}."

    cleaned: List[str] = []
    for item in distractors:
        if not isinstance(item, str) or not item.strip():
            return False, "All distractors must be non-empty strings."
        cleaned.append(item.strip())

    final_norm = normalize_text(final_diagnosis)
    distractor_norms = [normalize_text(item) for item in cleaned]

    if final_norm in distractor_norms:
        return False, "Final diagnosis appears in distractors."

    all_options = distractor_norms + [final_norm]
    if len(set(all_options)) != 4:
        return False, "All options (3 distractors + final diagnosis) must be unique."

    return True, "valid"


def build_user_prompt(
    case_prompt: str,
    diagnostic_reasoning: str,
    final_diagnosis: str,
    attempt: int,
    previous_failure: Optional[str],
) -> str:
    base = (
        f"case_prompt:\n{case_prompt}\n\n"
        f"diagnostic_reasoning:\n{diagnostic_reasoning}\n\n"
        f"final_diagnosis:\n{final_diagnosis}\n\n"
        "Return only JSON with key 'distractors' containing exactly 3 strings."
    )

    if attempt == 1:
        return base

    if attempt == 2:
        feedback = previous_failure or "Previous output did not satisfy validation."
        return (
            f"{base}\n\n"
            f"Validation feedback from previous attempt: {feedback}\n"
            "Fix the issue and return valid JSON only."
        )

    feedback = previous_failure or "Previous output did not satisfy validation."
    return (
        f"{base}\n\n"
        f"Validation feedback from previous attempt: {feedback}\n"
        "STRICT: Do not use the final diagnosis or synonyms/aliases of it. "
        "Ensure all 3 distractors are distinct and clinically plausible. "
        "Return JSON only."
    )


def resolve_records(limit: Optional[int]) -> Dataset:
    dataset_obj = load_dataset("zou-lab/MedCaseReasoning")

    if isinstance(dataset_obj, DatasetDict):
        if "test" in dataset_obj:
            records = dataset_obj["test"]
        else:
            first_split = next(iter(dataset_obj.keys()))
            records = dataset_obj[first_split]
    else:
        records = dataset_obj

    if limit is not None:
        take_n = min(limit, len(records))
        records = records.select(range(take_n))

    return records


def generate_distractors_for_case(
    llm: LLM,
    case_prompt: str,
    diagnostic_reasoning: str,
    final_diagnosis: str,
) -> Tuple[Optional[List[str]], Optional[str]]:
    previous_failure: Optional[str] = None

    for attempt in range(1, MAX_RETRIES + 1):
        temperature = TEMPERATURE_SCHEDULE[attempt - 1]
        sampling_params = SamplingParams(temperature=temperature, max_tokens=256)

        prompt = build_user_prompt(
            case_prompt=case_prompt,
            diagnostic_reasoning=diagnostic_reasoning,
            final_diagnosis=final_diagnosis,
            attempt=attempt,
            previous_failure=previous_failure,
        )

        messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        ]

        output = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
        text = output[0].outputs[0].text

        payload = extract_json(text)
        if payload is None:
            previous_failure = "Could not parse a valid JSON object."
            continue

        is_valid, reason = validate_distractors(final_diagnosis, payload.get("distractors"))
        if not is_valid:
            previous_failure = reason
            continue

        distractors = [item.strip() for item in payload["distractors"]]
        return distractors, None

    return None, previous_failure


def main(limit: Optional[int], output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = resolve_records(limit)

    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    success_count = 0
    skipped_count = 0
    skipped_case_ids: List[str] = []

    with out_path.open("w", encoding="utf-8") as handle:
        progress = tqdm(records, total=len(records), desc="Generating MCQ distractors")
        for row in progress:
            case_id = str(row.get("case_id") or row.get("id") or "unknown")
            case_prompt = (row.get("case_prompt") or "").strip()
            final_diagnosis = (row.get("final_diagnosis") or "").strip()
            diagnostic_reasoning = (row.get("diagnostic_reasoning") or "").strip()

            if not case_prompt or not final_diagnosis or not diagnostic_reasoning:
                skipped_count += 1
                skipped_case_ids.append(case_id)
                print(f"Skipping case_id={case_id}: missing required fields.")
                progress.set_postfix(success=success_count, skipped=skipped_count)
                continue

            distractors, failure_reason = generate_distractors_for_case(
                llm=llm,
                case_prompt=case_prompt,
                diagnostic_reasoning=diagnostic_reasoning,
                final_diagnosis=final_diagnosis,
            )

            if distractors is None:
                skipped_count += 1
                skipped_case_ids.append(case_id)
                print(
                    f"Skipping case_id={case_id}: failed after {MAX_RETRIES} attempts. "
                    f"Last failure: {failure_reason}"
                )
                progress.set_postfix(success=success_count, skipped=skipped_count)
                continue

            record = {
                "case_prompt": case_prompt,
                "final_diagnosis": final_diagnosis,
                "distractors": distractors,
                "original_diagnostic_reasoning": diagnostic_reasoning,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1
            progress.set_postfix(success=success_count, skipped=skipped_count)

    print("\nRun complete")
    print(f"Output path: {out_path}")
    print(f"Successful records: {success_count}")
    print(f"Skipped records: {skipped_count}")

    if skipped_case_ids:
        print("Skipped case_ids:")
        for item in skipped_case_ids:
            print(f"- {item}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate intermediate MCQ dataset from MedCaseReasoning")
    parser.add_argument("--limit", type=int, default=None, help="Number of entries to process")
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for output JSONL",
    )

    args = parser.parse_args()
    main(limit=args.limit, output_path=args.output_path)
