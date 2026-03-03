import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

DATASET = "zou-lab/MedCaseReasoning"
SPLIT = "val"
MODEL_NAME = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
DEFAULT_OUTPUT_PATH = "data/interim/medcasereasoning/formatted_diagnoses.jsonl"
DEFAULT_ERROR_LOG_PATH = "data/interim/medcasereasoning/formatting_errors.jsonl"
MAX_RETRIES = 3
TEMPERATURE_SCHEDULE = [0.1, 0.3, 0.5] # Kept very low because this is a deterministic formatting task
NUM_GPUS = 1

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
    # print("ATTEMPTING RETRY")
    return (
        f"{base}\n\n"
        f"Validation feedback from previous attempt: {feedback}\n"
        "STRICT: You must not change the actual words or characters. Only fix spaces and capitalization."
    )

def format_diagnosis(llm: LLM, original: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    previous_failure = None
    last_llm_output = None
    last_user_prompt = None

    for attempt in range(1, MAX_RETRIES + 1):
        temperature = TEMPERATURE_SCHEDULE[attempt - 1]
        sampling_params = SamplingParams(temperature=temperature, max_tokens=64)

        prompt = build_user_prompt(original, attempt, previous_failure)
        last_user_prompt = prompt

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        output = llm.chat([messages], sampling_params=sampling_params, use_tqdm=False)
        text = output[0].outputs[0].text
        last_llm_output = text

        payload = extract_json(text)
        if payload is None:
            previous_failure = "Could not parse a valid JSON object."
            continue

        is_valid, reason, cleaned = validate_formatting(
            original, payload.get("formatted_diagnosis")
        )
        
        if not is_valid:
            previous_failure = reason
            continue

        return cleaned, None, None, None

    return None, previous_failure, last_llm_output, last_user_prompt

def main(limit: Optional[int], output_path: str, error_log_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    error_log_path_obj = Path(error_log_path)
    error_log_path_obj.parent.mkdir(parents=True, exist_ok=True)

    records = load_dataset(DATASET, split=SPLIT)
    records = records.select(range(min(limit, len(records)))) if limit else records

    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=NUM_GPUS,
        max_model_len=2048
    )

    success_count = 0
    skipped_count = 0
    skipped_case_ids: List[str] = []

    with out_path.open("w", encoding="utf-8") as handle, \
         error_log_path_obj.open("w", encoding="utf-8") as error_handle:
        progress = tqdm(records, total=len(records), desc="Formatting Diagnoses")
        for row in progress:
            case_id = str(row.get("Unnamed: 0", "unknown"))
            original_diagnosis = row.get("final_diagnosis", "").strip()

            if not original_diagnosis:
                skipped_count += 1
                skipped_case_ids.append(case_id)
                continue

            formatted_diag, failure_reason, last_output, last_input = format_diagnosis(llm, original_diagnosis)

            if formatted_diag is None:
                skipped_count += 1
                skipped_case_ids.append(case_id)
                print(f"Skipping case_id={case_id}: failed after {MAX_RETRIES} attempts. Reason: {failure_reason}")
                
                error_record = {
                    "case_id": case_id,
                    "pmc_id": row.get("pmcid", ""),
                    "llm_input": last_input,
                    "failure_reason": failure_reason,
                    "llm_output": last_output
                }
                error_handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                continue

            record = {
                "pmc_id": row.get("pmcid", ""),
                "original_diagnosis": original_diagnosis,
                "formatted_diagnosis": formatted_diag
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1
            progress.set_postfix(success=success_count, skipped=skipped_count)

    print("\nRun complete.")
    print(f"Successful records: {success_count}")
    print(f"Failed records: {skipped_count}")
    if skipped_count > 0:
        print(f"Error log written to: {error_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format MedCaseReasoning final diagnoses")
    parser.add_argument("--limit", type=int, default=None, help="Number of entries to process")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--error_log_path", type=str, default=DEFAULT_ERROR_LOG_PATH, help="Path to write error logs")
    args = parser.parse_args()
    main(limit=args.limit, output_path=args.output_path, error_log_path=args.error_log_path)
