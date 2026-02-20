import json
import argparse
from vllm import LLM, SamplingParams

MODEL_PATH = "casperhansen/llama-3-8b-instruct-awq"
NUM_GPUS = 1 # GPUs available

SYSTEM_PROMPT = """You are an expert medical board exam writer.
Your task is to convert a clinical case report into a USMLE-style Multiple Choice Question.

Input:
A clinical case summary where the diagnosis is explicitly stated.

Instructions:
1. IDENTIFY the primary diagnosis.
2. REWRITE the case summary as a clinical vignette (patient presentation, history, symptoms).
   CRITICAL: You MUST obscure the diagnosis. Do not name the disease. Use phrases like "the condition" or "the defect".
3. GENERATE 3 plausible but incorrect differential diagnoses (distractors).
4. OUTPUT strictly in valid JSON format.

Output JSON Structure:
{
    "question": "The rewritten vignette...",
    "options": {
        "A": "Distractor 1",
        "B": "The Correct Diagnosis", 
        "C": "Distractor 2", 
        "D": "Distractor 3"
    },
    "answer_idx": "B" 
}
(Randomize the position of the correct answer).
"""

def format_prompt(entry):
    """Wraps the raw patient text in the instruction template."""
    user_content = f"Title: {entry['title']}\n\nPatient Case: {entry['patient']}"
    
    # Llama-3 specific formatting
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def main(input_file, output_file, limit=None):
    # Load Data
    print(f"Loading {input_file}...")
    entries = []
    with open(input_file, 'r') as f:
        for line in f:
            entries.append(json.loads(line))
            
    if limit: # Used for debugging
        print(f"Limit set: Processing only first {limit} entries.")
        entries = entries[:limit]

    # Prepare Prompts
    prompts = [format_prompt(e) for e in entries]

    # Initialize vLLM engine
    print("Initializing vLLM...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=NUM_GPUS,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        quantization="awq",
    )

    sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=1024,
        stop=["<|eot_id|>"] # Stop generating when done
    )

    # Generate (Batch Inference)
    print(f"Generating {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params)

    # Parse & Save
    success_count = 0
    with open(output_file, 'w') as f_out:
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            print(f"\nEntry {i} Generation:\n{generated_text}\n")
            original_entry = entries[i]
            
            try:
                # Attempt to parse JSON (Cleaning code fences if present)
                clean_json = generated_text.replace("```json", "").replace("```", "").strip()
                mcq_data = json.loads(clean_json)
                
                # Construct the Unified Schema Object
                final_entry = {
                    "id": f"pmc_gen_{original_entry['patient_uid']}",
                    "source": "pmc_patients_synthetic",
                    "specialty": original_entry.get("specialty", "Unclassified"),
                    "question": mcq_data["question"],
                    "options": mcq_data["options"],
                    "answer_idx": mcq_data["answer_idx"],
                    "meta_pmid": original_entry.get("PMID")
                }
                
                f_out.write(json.dumps(final_entry) + "\n")
                success_count += 1
                
            except json.JSONDecodeError:
                print(f"Failed to parse entry {i}")
                # Optional: Save failed generations to a log file to debug prompt
                
    print(f"\nDone. Successfully generated {success_count}/{len(entries)} MCQs.")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to stratified input file")
    parser.add_argument("--output", type=str, required=True, help="Path to output file")
    parser.add_argument("--limit", type=int, default=None, help="Debug: Run only N samples")
    args = parser.parse_args()

    main(args.input, args.output, args.limit)