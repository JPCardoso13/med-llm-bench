import json
import argparse
import random
import re
import uuid
from typing import List, Dict
from vllm import LLM, SamplingParams

MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

NUM_GPUS = 2

SYSTEM_PROMPT = """You are an expert medical board exam writer.
Your task is to create a high-quality Multiple Choice Question from a clinical case.

Input:
A patient case summary where the diagnosis is known.

Instructions:
1. Write a "Clinical Vignette" (question stem). 
   - Describe the patient's presentation, history, and symptoms.
   - CRITICAL: Do NOT mention the name of the diagnosis in the vignette. Obscure it.
2. Identify the "Correct Diagnosis".
3. Generate 3 "Distractors" (incorrect diagnoses).
   - They must be plausible differential diagnoses.
   - They must be distinct from the correct diagnosis.

Output:
Provide ONLY a raw JSON object with this exact structure:
{
    "question_stem": "The patient text...",
    "correct_answer": "The actual diagnosis",
    "distractors": ["Incorrect A", "Incorrect B", "Incorrect C"]
}
"""

def extract_json(text: str) -> Dict:
    """
    Extracts JSON object from text using Regex.
    Handles cases where models wrap output in markdown with ```json ... ```
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

def format_options(correct: str, distractors: List[str]) -> Dict:
    """
    Randomizes the order of correct and incorrect options and maps them to A, B, C, D.
    Returns: (options_dict, correct_key)
    """
    all_choices = [correct] + distractors

    random.shuffle(all_choices)

    keys = ["A", "B", "C", "D"]
    options_dict = {}
    correct_key = ""
    
    for i, choice in enumerate(all_choices):
        if i >= 4: # In case we have more than 4 options (shouldn't happen)
            break
        
        key = keys[i]
        options_dict[key] = choice
        
        if choice == correct:
            correct_key = key
            
    return options_dict, correct_key

def main(input_file, output_file, limit=None):
    print(f"Loading model: {MODEL_PATH}...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=NUM_GPUS,
        gpu_memory_utilization=0.9,
        quantization="awq" 
    )
    
    sampling_params = SamplingParams(
        temperature=0.3, # Creativity level
        top_p=0.9, # Keep diversity in distractors
        max_tokens=1024,
        stop=["<|eot_id|>"]
    )

    entries = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    if limit:
        entries = entries[:limit]

    prompts = []
    for entry in entries:
        user_content = f"Patient Case:\n{entry['patient']}\n\nPrimary Diagnosis:\n{entry.get('title', 'Unknown')}"
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        prompts.append(conversation)

    print(f"Generating MCQs for {len(prompts)} cases...")
    
    outputs = llm.chat(prompts, sampling_params)

    success_count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            original_entry = entries[i]
            
            data = extract_json(generated_text)
            
            if not data or "question_stem" not in data or "distractors" not in data:
                print(f"Failed to parse ID {i}: Invalid JSON structure")
                continue

            try:
                options, answer_idx = format_options(
                    data["correct_answer"], 
                    data["distractors"]
                )
            except Exception as e:
                print(f"Error formatting options for ID {i}: {e}")
                continue

            final_entry = {
                "id": f"pmc_gen_{str(uuid.uuid4())[:8]}",
                "question": data["question_stem"],
                "options": options,
                "answer_idx": answer_idx,
                "source": "pmc_patients_synthetic",
                "category": original_entry.get("specialty"),
            }
            
            f_out.write(json.dumps(final_entry) + "\n")
            success_count += 1

    print(f"Done. Successfully generated {success_count}/{len(entries)} MCQs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .jsonl file")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit")
    args = parser.parse_args()

    main(args.input, args.output, args.limit)
