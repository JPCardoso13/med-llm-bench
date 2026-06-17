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
DEFAULT_OUTPUT_PATH = "data/semi_processed/medcasemcq/fewshot/val_with_distractors.jsonl"
DEFAULT_FORMATTED_DIAGNOSES_PATH = "data/semi_processed/medcasemcq/fewshot/formatted_diagnoses_val.jsonl"
MAX_RETRIES = 3
TEMPERATURE_SCHEDULE = [0.7, 0.5, 0.3]
NUM_GPUS = 2

SYSTEM_PROMPT = """You are an expert clinical reasoning assistant.

Task:
Given a patient case prompt, diagnostic reasoning, and the final diagnosis, produce exactly 3 incorrect but clinically plausible differential diagnoses (distractors).

Rules:
1. Extract distractors explicitly mentioned in the diagnostic_reasoning if possible. Otherwise, infer plausible ones from the case_prompt.
2. Distractors must NOT be the same as, or aliases/synonyms of, the final diagnosis.
3. Output ONLY the core diagnostic entity. Do NOT include underlying causes, mechanisms, or compound conditions.
    - BAD: "Acute Kidney Injury due to Contrast-Induced Nephropathy" -> GOOD: "Acute kidney injury"
    - BAD: "Myocardial Infarction and Hyperkalemia-induced Bradyarrhythmia" -> GOOD: "Myocardial infarction"
4. Use standard medical Sentence case. Capitalize only the first letter of the diagnosis, proper nouns (eponyms), and standard acronyms.
    - BAD: "congestive heart failure" -> GOOD: "Congestive heart failure"
    - BAD: "lyme disease" -> GOOD: "Lyme disease"
    - BAD: "Acute Pancreatitis" -> GOOD: "Acute pancreatitis"
    - BAD: "hiv infection" -> GOOD: "HIV infection"

Return ONLY valid JSON in this exact format:
{"distractors": ["d1", "d2", "d3"]}

/no_think
"""


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def extract_json(text: str) -> Optional[Dict[str, object]]:
    text_without_thoughts = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    match = re.search(r"\{.*\}", text_without_thoughts, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_and_clean_distractors(final_diagnosis: str, distractors: object) -> Tuple[bool, str, Optional[List[str]]]:
    if not isinstance(distractors, list):
        return False, "Missing or invalid 'distractors' list.", None

    if len(distractors) != 3:
        return False, f"Expected exactly 3 distractors, got {len(distractors)}.", None

    cleaned: List[str] = []
    for item in distractors:
        if not isinstance(item, str) or not item.strip():
            return False, "All distractors must be non-empty strings.", None
        cleaned.append(item.strip())

    final_norm = normalize_text(final_diagnosis)
    distractor_norms = [normalize_text(item) for item in cleaned]

    if final_norm in distractor_norms:
        return False, "Final diagnosis appears in distractors.", None

    all_options = distractor_norms + [final_norm]
    if len(set(all_options)) != 4:
        return False, "All options (3 distractors + final diagnosis) must be unique.", None

    return True, "valid", cleaned


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


def process_batch(llm: LLM, records: List[dict], attempt: int) -> Tuple[List[dict], List[dict]]:
    temperature = TEMPERATURE_SCHEDULE[attempt - 1]
    sampling_params = SamplingParams(temperature=temperature, max_tokens=512)

    messages = []
    for row in records:
        prompt = build_user_prompt(
            case_prompt=row["case_prompt"],
            diagnostic_reasoning=row["diagnostic_reasoning"],
            final_diagnosis=row["final_diagnosis"],
            attempt=attempt,
            previous_failure=row.get("failure_reason"),
        )
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

        is_valid, reason, cleaned = validate_and_clean_distractors(
            row["final_diagnosis"], payload.get("distractors")
        )
        if not is_valid:
            row["failure_reason"] = reason
            failed.append(row)
            continue

        row["distractors"] = cleaned
        row.pop("failure_reason", None)
        row.pop("_last_user_prompt", None)
        row.pop("_last_llm_output", None)
        successful.append(row)

    return successful, failed


def main(limit: Optional[int], split: str, formatted_diagnoses_path: str, output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    formatted_diagnoses_path = Path(formatted_diagnoses_path)
    diagnosis_mapping: Dict[str, str] = {}
    with formatted_diagnoses_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                diagnosis_mapping[entry["pmc_id"]] = entry["formatted_diagnosis"]
    print(f"Loaded {len(diagnosis_mapping)} formatted diagnoses from {formatted_diagnoses_path}")

    dataset = load_dataset(DATASET, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    records = []
    skipped_count = 0
    for row in dataset:
        case_id = str(row.get("Unnamed: 0", "unknown"))
        pmc_id = row.get("pmcid", "").strip()
        case_prompt = row.get("case_prompt", "").strip()
        final_diagnosis = diagnosis_mapping.get(pmc_id, "").strip()
        diagnostic_reasoning = row.get("diagnostic_reasoning", "").strip()

        if not case_prompt or not final_diagnosis or not diagnostic_reasoning:
            skipped_count += 1
            print(f"Skipping case_id={case_id}: missing required fields.")
            continue

        records.append({
            "case_id": case_id,
            "pmc_id": pmc_id,
            "article_link": row.get("article_link", ""),
            "text": row.get("text", ""),
            "case_prompt": case_prompt,
            "final_diagnosis": final_diagnosis,
            "diagnostic_reasoning": diagnostic_reasoning,
        })

    print(f"Loaded {len(records)} records ({skipped_count} skipped due to missing fields).")

    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.85,
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
                "article_link": row["article_link"],
                "text": row["text"],
                "case_prompt": row["case_prompt"],
                "final_diagnosis": row["final_diagnosis"],
                "distractors": row["distractors"],
                "diagnostic_reasoning": row["diagnostic_reasoning"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nRun complete")
    print(f"Output path: {out_path}")
    print(f"Successful records: {len(all_successful)}")
    print(f"Skipped records (missing fields): {skipped_count}")
    print(f"Failed after {MAX_RETRIES} retries: {len(current_batch)}")

    if current_batch:
        print("Failed case_ids:")
        for row in current_batch:
            print(f"- {row['case_id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate intermediate MCQ dataset from MedCaseReasoning")
    parser.add_argument("--limit", type=int, default=None, help="Number of entries to process")
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT, help="HuggingFace dataset split to use")
    parser.add_argument("--formpath", type=str, default=DEFAULT_FORMATTED_DIAGNOSES_PATH, help="Path to formatted diagnoses JSONL file")
    parser.add_argument("--outpath", type=str, default=DEFAULT_OUTPUT_PATH, help="Path for output JSONL file")

    args = parser.parse_args()
    main(limit=args.limit, split=args.split, formatted_diagnoses_path=args.formpath, output_path=args.outpath)
