import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from vllm import LLM, SamplingParams

DATASET = "zou-lab/MedCaseReasoning"
DEFAULT_SPLIT = "val"
MODEL_NAME = "Qwen/Qwen3-32B"
DEFAULT_OUTPUT_PATH = "data/semi_processed/medcasemcq/fewshot/formatted_diagnoses_val.jsonl"
DEFAULT_ERROR_LOG_PATH = "logs/dataset_creation/medcasemcq/fewshot/formatting_errors.jsonl"
MAX_RETRIES = 3
TEMPERATURE_SCHEDULE = [0.1, 0.3, 0.5] # Kept very low because this is a deterministic formatting task
NUM_GPUS = 2

SYSTEM_PROMPT = """You are a strict medical text formatting assistant.

Task:
Convert the provided raw medical diagnosis string into standard medical Sentence case.

Rules:
1. Capitalize ONLY the first letter of the diagnosis, proper nouns (eponyms like "Crohn's"), and standard acronyms (like "HIV", "PFAPA").
2. Replace underscores with spaces. Split CamelCase where appropriate.
3. DO NOT change, add, or remove any words. DO NOT expand acronyms.
4. If the input is already perfectly formatted, return the EXACT SAME STRING. Do not over-correct.

Examples:
- Input: "AmeloblasticFibroma" -> Output: "Ameloblastic fibroma"
- Input: "Scleroderma_renal_crisis" -> Output: "Scleroderma renal crisis"
- Input: "PFAPAsyndrome" -> Output: "PFAPA syndrome"
- Input: "congestive heart failure" -> Output: "Congestive heart failure"
- Input: "Acute kidney injury" -> Output: "Acute kidney injury"

Return ONLY valid JSON in this exact format:
{"formatted_diagnosis": "String"}

/no_think
"""

def extract_json(text: str) -> Optional[Dict[str, object]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def strip_for_compare(text: str) -> str:
    """Removes all non-alphanumeric characters (spaces, punctuation, apostrophes) and lowers case."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def validate_formatting(original: str, formatted: object) -> Tuple[bool, str, Optional[str]]:
    if not isinstance(formatted, str) or not formatted.strip():
        return False, "Missing or invalid 'formatted_diagnosis' string.", None

    cleaned = formatted.strip()

    orig_stripped = strip_for_compare(original)
    form_stripped = strip_for_compare(cleaned)

    if orig_stripped != form_stripped:
        return False, f"Hallucination detected. Core characters changed. Expected base '{orig_stripped}', got '{form_stripped}'. Do not add, remove, or alter words.", None

    return True, "valid", cleaned

def build_user_prompt(original: str, attempt: int, previous_failure: Optional[str]) -> str:
    base = f"Raw Diagnosis Input:\n{original}\n\nReturn ONLY JSON with key 'formatted_diagnosis'."

    if attempt == 1:
        return base

    feedback = previous_failure or "Previous output did not satisfy validation."
    return (
        f"{base}\n\n"
        f"Validation feedback from previous attempt: {feedback}\n"
        "STRICT: You must not change the actual words or characters. Only fix spaces and capitalization."
    )

def process_batch(llm: LLM, records: List[dict], attempt: int) -> Tuple[List[dict], List[dict]]:
    temperature = TEMPERATURE_SCHEDULE[attempt - 1]
    sampling_params = SamplingParams(temperature=temperature, max_tokens=64)

    messages = []
    for row in records:
        prompt = build_user_prompt(row["original_diagnosis"], attempt, row.get("failure_reason"))
        row["_last_user_prompt"] = prompt
        messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

    outputs = llm.chat(messages, sampling_params=sampling_params)

    successful = []
    failed = []

    for row, output in zip(records, outputs):
        text = output.outputs[0].text
        row["_last_llm_output"] = text

        payload = extract_json(text)
        if payload is None:
            row["failure_reason"] = "Could not parse a valid JSON object."
            failed.append(row)
            continue

        is_valid, reason, cleaned = validate_formatting(
            row["original_diagnosis"], payload.get("formatted_diagnosis")
        )

        if not is_valid:
            row["failure_reason"] = reason
            failed.append(row)
            continue

        row["formatted_diagnosis"] = cleaned
        row.pop("failure_reason", None)
        row.pop("_last_user_prompt", None)
        row.pop("_last_llm_output", None)
        successful.append(row)

    return successful, failed

def main(limit: Optional[int], split: str, output_path: str, error_log_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path_obj = Path(error_log_path)
    error_log_path_obj.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records = []
    skipped_empty = 0
    for row in dataset:
        case_id = str(row.get("Unnamed: 0", "unknown"))
        original = row.get("final_diagnosis", "").strip()
        if not original:
            skipped_empty += 1
            print(f"Skipping case_id={case_id}: empty diagnosis.")
            continue
        records.append({
            "case_id": case_id,
            "pmc_id": row.get("pmcid", ""),
            "original_diagnosis": original,
        })

    print(f"Loaded {len(records)} records ({skipped_empty} skipped, empty diagnosis).")

    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        tensor_parallel_size=NUM_GPUS,
        distributed_executor_backend="ray",
        max_model_len=4096
    )

    all_successful = []
    current_batch = records

    for attempt in range(1, MAX_RETRIES + 1):
        if not current_batch:
            break
        print(f"\n--- Attempt {attempt}/{MAX_RETRIES} (Processing {len(current_batch)} record(s)) ---")
        success, failed = process_batch(llm, current_batch, attempt)
        all_successful.extend(success)
        current_batch = failed
        print(f"Attempt {attempt}: Success={len(success)}, Failed={len(failed)}")

    with out_path.open("w", encoding="utf-8") as handle:
        for row in all_successful:
            record = {
                "pmc_id": row["pmc_id"],
                "original_diagnosis": row["original_diagnosis"],
                "formatted_diagnosis": row["formatted_diagnosis"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with error_log_path_obj.open("w", encoding="utf-8") as error_handle:
        for row in current_batch:
            error_record = {
                "case_id": row["case_id"],
                "pmc_id": row["pmc_id"],
                "llm_input": row.get("_last_user_prompt", ""),
                "failure_reason": row.get("failure_reason", ""),
                "llm_output": row.get("_last_llm_output", ""),
            }
            error_handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")

    print("\nRun complete.")
    print(f"Successful records: {len(all_successful)}")
    print(f"Failed records: {len(current_batch)}")
    print(f"Empty diagnosis skipped: {skipped_empty}")
    if current_batch:
        print(f"Error log written to: {error_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format MedCaseReasoning final diagnoses")
    parser.add_argument("--limit", type=int, default=None, help="Number of entries to process")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="HuggingFace dataset split to use")
    parser.add_argument("--outpath", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--errpath", type=str, default=DEFAULT_ERROR_LOG_PATH, help="Path to write error logs")
    args = parser.parse_args()
    main(limit=args.limit, split=args.split, output_path=args.outpath, error_log_path=args.errpath)
