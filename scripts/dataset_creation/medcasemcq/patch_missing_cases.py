import json
from datasets import load_dataset

# Configuration
DATASET = "zou-lab/MedCaseReasoning"
SPLIT = "test"

# Dictionary mapping Case ID ("Unnamed: 0") to a list of manual distractors
MANUAL_PATCHES = {
    "2943": ["Paraneoplastic pemphigus", "Pyoderma gangrenosum", "Herpetic whitlow"],
}

# Paths
OUTPUT_PATH = "data/interim/medcasereasoning/mcq_dataset.jsonl"
FORMATTED_DIAGNOSES_PATH = "data/interim/medcasereasoning/formatted_diagnoses_test.jsonl"

def patch_missing_cases():
    if not MANUAL_PATCHES:
        print("No cases to patch in MANUAL_PATCHES.")
        return

    print(f"Loading formatted diagnoses from {FORMATTED_DIAGNOSES_PATH}...")
    formatted_diagnoses = {}
    with open(FORMATTED_DIAGNOSES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                formatted_diagnoses[entry["pmc_id"]] = entry["formatted_diagnosis"]

    print(f"Loading dataset '{DATASET}' (split: {SPLIT})...")
    records = load_dataset(DATASET, split=SPLIT)
    
    pending_patches = set(MANUAL_PATCHES.keys())
    patched_count = 0

    print(f"Appending patched cases to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "a", encoding="utf-8") as handle:
        for row in records:
            case_id = str(row.get("Unnamed: 0", ""))
            
            if case_id in pending_patches:
                pmc_id = row.get("pmcid", "").strip()
                
                final_diagnosis = formatted_diagnoses.get(pmc_id)
                if not final_diagnosis:
                    print(f"  Warning: Formatted diagnosis not found for {pmc_id} (Case ID: {case_id}). Using raw.")
                    final_diagnosis = row.get("final_diagnosis", "").strip()

                record = {
                    "pmc_id": pmc_id,
                    "article_link": row.get("article_link", ""),
                    "text": row.get("text", ""),
                    "case_prompt": row.get("case_prompt", "").strip(),
                    "final_diagnosis": final_diagnosis,
                    "distractors": MANUAL_PATCHES[case_id],
                    "diagnostic_reasoning": row.get("diagnostic_reasoning", "").strip(),
                }

                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                pending_patches.remove(case_id)
                patched_count += 1
                print(f"  -> Successfully patched Case ID: {case_id} (PMC ID: {pmc_id})")

                if not pending_patches:
                    break

    print(f"\nPatching complete. {patched_count} cases appended.")
    
    if pending_patches:
        print(f"WARNING: The following Case IDs were not found in the dataset: {pending_patches}")

if __name__ == "__main__":
    patch_missing_cases()
