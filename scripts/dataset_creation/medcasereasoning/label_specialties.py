import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams

INPUT_PATH = "data/processed/mcr_mcq/mcr_mcq.jsonl"
OUTPUT_PATH = "data/processed/mcr_mcq/mcr_mcq_labeled.jsonl"
MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
NUM_GPUS = 2
MAX_RETRIES = 3
TEMPERATURE_SCHEDULE = [0.7, 0.5, 0.3]

SPECIALTIES_LIST = [
    "Cardiology",
    "Pulmonology",
    "Gastroenterology / Hepatology",
    "Nephrology",
    "Endocrinology",
    "Rheumatology",
    "Hematology",
    "Oncology",
    "Infectious Disease",
    "Neurology",
    "Dermatology",
    "Immunology / Allergy",
    "Medical Genetics",
    "Pediatrics",
    "Psychiatry",
    "Obstetrics / Gynecology",
    "Toxicology",
    "Urology",
    "Surgery",
    "General Internal Medicine"
]

SYSTEM_PROMPT = f"""You are an expert medical classification assistant.

Task:
Classify the given medical case into 1 to 2 applicable medical specialties.

Rules:
1. You MUST choose from the following exact list of specialties:
{json.dumps(SPECIALTIES_LIST, indent=2)}
2. Anchor your classification STRICTLY to the 'Correct Diagnosis'. Do NOT tag specialties based on misdiagnoses or initial consultations.
3. Be conservative. Output exactly 1 to 2 specialties. 
4. DO NOT hallucinate specialties like 'Ophthalmology' or 'Orthopedics' if they aren't on the list. Map them to the closest systemic specialty or General Internal Medicine.
5. Return ONLY valid JSON in this exact format:
{{"specialties": ["Specialty 1"]}}

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

def label_case(llm: LLM, question: str, correct_diagnosis: str) -> Tuple[List[str], bool]:
    """Returns a tuple of (specialties_list, is_defaulted)"""
    previous_failure = None

    for attempt in range(1, MAX_RETRIES + 1):
        temperature = TEMPERATURE_SCHEDULE[attempt - 1]
        sampling_params = SamplingParams(temperature=temperature, max_tokens=128)
        
        user_content = f"Case Prompt:\n{question}\n\nCorrect Diagnosis:\n{correct_diagnosis}\n\nReturn JSON only."
        if previous_failure:
            user_content += f"\n\nValidation failed from previous attempt: {previous_failure}. Fix this and pick strictly from the list."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        output = llm.chat([messages], sampling_params=sampling_params, use_tqdm=False)
        text = output[0].outputs[0].text

        if attempt > 1:
            print(f"\n[DEBUG - Attempt {attempt}]")
            print(f"LLM RESPONSE:\n{repr(text)}")

        payload = extract_json(text)
        if payload is None:
            previous_failure = "Could not parse JSON."
            continue

        is_valid, reason, valid_specialties = validate_specialties(payload)
        if not is_valid:
            previous_failure = reason
            continue

        return valid_specialties, False

    # Fallback if it completely fails after 3 attempts
    return ["General Internal Medicine"], True

def main(limit: Optional[int], output_path: str) -> None:
    input_file = Path(INPUT_PATH)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: {INPUT_PATH} not found.")
        return

    print("Initializing LLM engine...")
    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.85,
        #enforce_eager=True,
        tensor_parallel_size=NUM_GPUS,
        distributed_executor_backend="ray",
        max_model_len=4096
    )

    with input_file.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    
    if limit:
        records = records[:limit]

    success_count = 0
    defaulted_count = 0

    with output_file.open("w", encoding="utf-8") as out:
        progress = tqdm(records, total=len(records), desc="Classifying Specialties")
        for row in progress:
            correct_diag = row["options"][row["answer_idx"]]
            
            specialties, is_defaulted = label_case(llm, row["question"], correct_diag)
            row["specialties"] = specialties
            
            if is_defaulted:
                defaulted_count += 1
            else:
                success_count += 1
                
            progress.set_postfix(success=success_count, defaulted=defaulted_count)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nRun complete")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Successful records: {success_count}")
    print(f"Defaulted records: {defaulted_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label medical cases with medical specialties")
    parser.add_argument("--limit", type=int, default=None, help="Number of entries to process")
    parser.add_argument(
        "--output_path",
        type=str,
        default=OUTPUT_PATH,
        help="Path for output JSONL file",
    )

    args = parser.parse_args()
    main(limit=args.limit, output_path=args.output_path)