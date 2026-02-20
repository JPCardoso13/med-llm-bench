import json
import argparse
import random
import re
import uuid
from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams

MODEL_PATH = "Qwen/Qwen3-32B-AWQ"
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are an expert medical board exam writer.
Your task is to create a high-quality Multiple Choice Question from a clinical case.

Input:
A patient case summary. The primary diagnosis is provided to you.

Instructions:
1. Write a "Clinical Vignette" (question stem). 
   - Describe the presentation, history, and symptoms.
   - CRITICAL: Do NOT mention the name of the diagnosis (or close variations) in the vignette.
2. Output the exact "Correct Diagnosis".
3. Generate 3 "Distractors" (incorrect, but plausible differential diagnoses).

Output strictly as a JSON object:
{
    "question_stem": "...",
    "correct_answer": "...",
    "distractors": ["...", "...", "..."]
}
"""

def extract_json(text: str) -> Dict:
    """
    Extracts JSON object from text.
    """
    # This regex looks for the first '{' and the last '}'
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        # Fallback: assume the whole text might be JSON
        json_str = text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def hard_validation(data: Dict) -> Tuple[bool, str]:
    """Checks structure and types. Python-level validation."""
    if not data:
        return False, "Failed to parse JSON."
    if "question_stem" not in data or not isinstance(data["question_stem"], str):
        return False, "Missing or invalid 'question_stem'."
    if "correct_answer" not in data or not isinstance(data["correct_answer"], str):
        return False, "Missing or invalid 'correct_answer'."
    if "distractors" not in data or not isinstance(data["distractors"], list):
        return False, "Missing or invalid 'distractors'."
    if len(data["distractors"]) != 3:
        return False, f"Expected 3 distractors, got {len(data['distractors'])}."
    
    # Check for empty strings
    if not data["question_stem"].strip() or not data["correct_answer"].strip():
         return False, "Empty strings found in required fields."
         
    return True, "Valid"

def soft_validation_leakage(data: Dict) -> bool:
    """
    Basic NLP check to see if the correct answer leaked into the question stem.
    Returns True if SAFE (no leakage), False if LEAKED.
    """
    stem = data["question_stem"].lower()
    answer = data["correct_answer"].lower()
    
    # If the exact answer string is in the stem, it's an automatic fail.
    # Note: For a true LLM Judge, you would prompt a model here. 
    # For speed during generation, string matching catches 90% of lazy generation.
    if answer in stem:
        return False
    return True

def format_options(correct: str, distractors: List[str]) -> Tuple[Dict, str]:
    """Randomizes options and assigns A, B, C, D."""
    all_choices = [correct] + distractors
    random.shuffle(all_choices)
    
    keys = ["A", "B", "C", "D"]
    options_dict = {}
    correct_key = ""
    
    for i, choice in enumerate(all_choices[:4]):
        key = keys[i]
        options_dict[key] = choice
        if choice == correct:
            correct_key = key
            
    return options_dict, correct_key

def main(input_file, output_file, limit=None):
    print(f"Loading vLLM model: {MODEL_PATH}...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        quantization="awq"
    )
    
    # Slightly higher temperature for generation
    sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)

    entries = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip(): entries.append(json.loads(line))
    if limit: entries = entries[:limit]

    success_count = 0
    discard_count = 0

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry_idx, entry in enumerate(entries):
            user_content = f"Patient Case:\n{entry['patient']}\n\nPrimary Diagnosis:\n{entry.get('title', 'Unknown')}"
            prompts = [[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]]

            # RETRY LOOP
            attempt = 0
            success = False
            
            while attempt < MAX_RETRIES and not success:
                attempt += 1
                
                # Note: vLLM chat expects a list of prompt lists. We pass one at a time here 
                # for the retry logic, though batching is faster if implemented complexly.
                output = llm.chat(prompts, sampling_params, use_tqdm=False)
                generated_text = output[0].outputs[0].text
                
                data = extract_json(generated_text)
                
                # 1. Hard Validation (Structure)
                is_valid, reason = hard_validation(data)
                if not is_valid:
                    print(f"Entry {entry_idx} | Attempt {attempt} Failed: {reason}")
                    continue # Try again
                
                # 2. Soft Validation (Leakage)
                if not soft_validation_leakage(data):
                    print(f"Entry {entry_idx} | Attempt {attempt} Failed: Answer leaked into stem.")
                    continue # Try again
                    
                # If we passed both, we succeed!
                success = True
                
                # Format and Save
                options, answer_idx = format_options(data["correct_answer"], data["distractors"])
                
                final_entry = {
                    "id": f"pmc_gen_{str(uuid.uuid4())[:8]}",
                    "question": data["question_stem"],
                    "options": options,
                    "answer_idx": answer_idx,
                    "source": "pmc_patients_synthetic",
                    "category": entry.get("specialty", "General"),
                    "tags": {
                        "original_pmid": entry.get("PMID", ""),
                        "generation_attempts": attempt
                    }
                }
                
                f_out.write(json.dumps(final_entry) + "\n")
                success_count += 1
                print(f"Entry {entry_idx} | Success on attempt {attempt}")

            if not success:
                print(f"Entry {entry_idx} | DISCARDED after {MAX_RETRIES} attempts.")
                discard_count += 1

    print(f"\n--- Generation Complete ---")
    print(f"Successfully generated: {success_count}")
    print(f"Discarded (Failed Validation): {discard_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    main(args.input, args.output, args.limit)