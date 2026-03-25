import json
import random
import re
from pathlib import Path

INPUT_PATH = "data/interim/medcasereasoning/mcq_dataset_test.jsonl"
OUTPUT_PATH = "data/processed/mcr_mcq/mcr_mcq.jsonl"

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

def normalize_casing_leaks(text: str) -> str:
    """
    Forces 'Syndrome' to lowercase unless is is the first word.
    This is an edge case that was missed in the final diagnosis formatting step.
    """
    text = re.sub(r'(?<=\w\s)Syndrome\b', 'syndrome', text)
    return text

def build_dataset():
    input_file = Path(INPUT_PATH)
    output_file = Path(OUTPUT_PATH)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    success_count = 0
    
    random.seed(13)

    with input_file.open("r", encoding="utf-8") as infile, \
         output_file.open("w", encoding="utf-8") as outfile:
        
        for id_counter, line in enumerate(infile):
            if not line.strip():
                continue
            
            row = json.loads(line)
            
            # Normalize casing
            correct_diagnosis = normalize_casing_leaks(row["final_diagnosis"])
            distractors = [normalize_casing_leaks(d) for d in row["distractors"]]
            
            # Mix and label options A, B, C, D
            all_options = distractors + [correct_diagnosis]
            random.shuffle(all_options)
            
            options_dict = {chr(65 + i): opt for i, opt in enumerate(all_options)}
            
            # Find the correct answer index
            answer_idx = next(k for k, v in options_dict.items() if v == correct_diagnosis)
            
            # Construct the final question
            stem = random.choice(QUESTION_STEMS)
            full_question = f"{row['case_prompt']}\n\n{stem}"

            # Build the final flat record
            record = {
                "id": str(id_counter),
                "question": full_question,
                "options": options_dict,
                "answer_idx": answer_idx,
                "pmc_id": row.get("pmc_id", ""),
                "article_link": row.get("article_link", "")
            }
            
            # Write to output
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            success_count += 1

    print("\n--- Dataset Assembly Complete ---")
    print(f"Output saved to: {OUTPUT_PATH}")
    print(f"Total records processed: {success_count}")

if __name__ == "__main__":
    build_dataset()
