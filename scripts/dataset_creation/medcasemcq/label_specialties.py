import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from vllm import LLM, SamplingParams


INPUT_PATH = "data/semi_processed/medcasemcq/eval_no_specialties.jsonl"
OUTPUT_PATH = "data/processed/medcasemcq/eval.jsonl"
DEFAULT_ERROR_LOG_PATH = "logs/dataset_creation/medcasemcq/specialty_labeling_errors.jsonl"

MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
NUM_GPUS = 2
MAX_RETRIES = 3
TEMPERATURE_SCHEDULE = [0.1, 0.3, 0.5] 

SPECIALTIES_LIST = [
    "Cardiology", "Pulmonology", "Gastroenterology / Hepatology", "Nephrology",
    "Endocrinology", "Rheumatology", "Hematology", "Oncology", "Infectious Disease",
    "Neurology", "Dermatology", "Immunology / Allergy", "Medical Genetics", "Pediatrics",
    "Psychiatry", "Obstetrics / Gynecology", "Toxicology", "Urology", "General Surgery",
    "Ophthalmology", "Otolaryngology", "Orthopedics", "General Internal Medicine"
]

SYSTEM_PROMPT = f"""You are an expert medical classification assistant.

Task:
Classify the given medical case into 1 to 2 applicable medical specialties.

Rules:
1. You MUST choose from the following exact list of specialties:
{json.dumps(SPECIALTIES_LIST, indent=2)}
2. Anchor your classification STRICTLY to the 'Diagnosis'.
3. Do not tag specialties based on ruled-out differential diagnoses or negative laboratory tests.
4. Account for patient demographics: if the patient is *under* 18 years old, include 'Pediatrics'.
5. Be conservative. Output exactly 1 to 2 specialties. 
6. Return ONLY valid JSON in the exact given format.

Example Input:
Case Prompt: A 6-year-old boy presents with right lower quadrant pain...
Diagnosis: Acute Appendicitis

Example Output:
{{"specialties": ["Pediatrics", "General Surgery"]}}

/no_think
"""


def extract_json(text: str) -> Optional[Dict[str, object]]:
    text_without_thoughts = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text_without_thoughts, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_specialties(payload: object) -> Tuple[bool, str, Optional[List[str]]]:
    if not isinstance(payload, dict) or "specialties" not in payload:
        return False, "Missing 'specialties' key.", None
    
    specs = payload["specialties"]
    if not isinstance(specs, list) or len(specs) == 0 or len(specs) > 2:
        return False, "Must provide a list of 1 to 2 specialties.", None
        
    cleaned = []
    for s in specs:
        if s not in SPECIALTIES_LIST:
            return False, f"Hallucinated specialty: '{s}'. Must be from the approved list.", None
        cleaned.append(s)
        
    cleaned = list(dict.fromkeys(cleaned))
    return True, "valid", cleaned


def process_batch(llm: LLM, records: List[dict], attempt: int) -> Tuple[List[dict], List[dict]]:
    temperature = TEMPERATURE_SCHEDULE[attempt - 1]
    sampling_params = SamplingParams(temperature=temperature, max_tokens=128)

    messages = []
    for row in records:
        diagnosis = row["options"][row["answer_idx"]]
        user_content = f"Case Prompt:\n{row['question']}\n\nDiagnosis:\n{diagnosis}\n\nReturn JSON only."
        
        if "failure_reason" in row:
            user_content += f"\n\nPrevious attempt failed validation: {row['failure_reason']}. Avoid this mistake."
        row["_last_user_prompt"] = user_content
            
        messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ])

    outputs = llm.chat(messages, sampling_params=sampling_params)
    
    successful_records = []
    failed_records = []
    
    for row, output in zip(records, outputs):
        text = output.outputs[0].text
        row["_last_llm_output"] = text
        payload = extract_json(text)
        
        if payload is None:
            row["failure_reason"] = "Could not parse JSON."
            failed_records.append(row)
            continue
            
        is_valid, fail_reason, specialties = validate_specialties(payload)
        if not is_valid:
            row["failure_reason"] = fail_reason
            failed_records.append(row)
            continue
            
        row["specialties"] = specialties
        row.pop("failure_reason", None) # Present if the attempt is a retry
        row.pop("_last_user_prompt", None)
        row.pop("_last_llm_output", None)
        successful_records.append(row)
        
    return successful_records, failed_records


def main(limit: Optional[int], output_path: str, error_log_path: str) -> None:
    input_file = Path(INPUT_PATH)
    output_file = Path(output_path)
    error_log_file = Path(error_log_path)
    
    if not input_file.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_log_file.parent.mkdir(parents=True, exist_ok=True)

    print("Initializing LLM engine...")
    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        tensor_parallel_size=NUM_GPUS,
        distributed_executor_backend="ray",
        max_model_len=4096
    )

    with input_file.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    if limit:
        records = records[:limit]

    all_successful = []
    current_batch = records
    
    for attempt in range(1, MAX_RETRIES + 1):
        if not current_batch:
            break
        print(f"\n--- Attempt {attempt}/{MAX_RETRIES} (Processing {len(current_batch)} record(s)) ---")
        success, failed = process_batch(llm, current_batch, attempt)
        all_successful.extend(success)
        current_batch = failed  # The failed records become the batch for the next attempt
        print(f"Finished attempt {attempt}:\nSuccess = {len(success)}\nFailed = {len(failed)}")
        
    # Handle absolute failures (Defaulting)
    hard_failures = []
    defaulted_count = len(current_batch)
    
    for row in current_batch:
        hard_failures.append(
            {
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "failure_reason": row.get("failure_reason", ""),
                "llm_input": row.get("_last_user_prompt", ""),
                "llm_output": row.get("_last_llm_output", ""),
            }
        )
        
        row["specialties"] = ["General Internal Medicine"]
        row.pop("failure_reason", None)  # Leftover from last attempt
        row.pop("_last_user_prompt", None)
        row.pop("_last_llm_output", None)
        all_successful.extend([row])

    with output_file.open("w", encoding="utf-8") as out:
        for row in all_successful:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    with error_log_file.open("w", encoding="utf-8") as err:
        for error_record in hard_failures:
            err.write(json.dumps(error_record, ensure_ascii=False) + "\n")

    print("\nRun complete")
    print(f"Output path: {output_path}")
    print(f"Error log path: {error_log_path}")
    print(f"Successful records: {len(all_successful) - defaulted_count}")
    print(f"Defaulted records: {defaulted_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label medical case questions with medical specialties")
    parser.add_argument("--limit", type=int, default=None, help="Number of entries to process")
    parser.add_argument(
        "--output_path",
        type=str,
        default=OUTPUT_PATH,
        help="Path for output JSONL file",
    )
    parser.add_argument(
        "--error_log_path",
        type=str,
        default=DEFAULT_ERROR_LOG_PATH,
        help="Path to write hard-failure error logs",
    )

    args = parser.parse_args()
    main(limit=args.limit, output_path=args.output_path, error_log_path=args.error_log_path)
