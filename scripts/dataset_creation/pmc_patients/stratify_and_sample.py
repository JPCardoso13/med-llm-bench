import json
import random
from collections import Counter, defaultdict

SPECIALTY_RULES = {
    "Cardiology": ["Heart", "Cardiac", "Coronary", "Valve", "Arrhythmia", "Atrial", "Ventricular", "Myocardial", "Pericardial", "Aortic"],
    "Neurology": ["Brain", "Nerve", "Spinal", "Seizure", "Epilepsy", "Stroke", "Cerebral", "Neuropathy", "Alzheimer", "Parkinson", "Central Nervous System"],
    "Oncology": ["Neoplasms", "Tumor", "Cancer", "Carcinoma", "Sarcoma", "Leukemia", "Lymphoma", "Metastasis", "Chemotherapy"],
    "Pediatrics": ["Child", "Infant", "Newborn", "Adolescent", "Pediatrics", "Congenital"],
    "Gastroenterology": ["Liver", "Gastric", "Intestine", "Colonic", "Pancreatitis", "Hepatitis", "Abdominal", "Gastrointestinal"],
    "Pulmonology": ["Lung", "Pulmonary", "Respiratory", "Bronchial", "Pneumonia", "Asthma", "Pleural"]
}

INPUT_FILE = "data/intermediate/pmc_patients/pmc_patients_with_mesh.jsonl"
OUTPUT_FEWSHOT = "data/intermediate/pmc_patients/pmc_fewshot.jsonl"
OUTPUT_QUESTION = "data/intermediate/pmc_patients/pmc_questions.jsonl"

TARGET_PER_CLASS = 400 # Number of samples per specialty

def get_specialty(mesh_terms):
    scores = Counter()

    for term in mesh_terms:
        term_l = term.lower()
        for specialty, keywords in SPECIALTY_RULES.items():
            for k in keywords:
                if k.lower() in term_l:
                    scores[specialty] += 1
                    break  # Prevent double counting for the same specialty

    if scores:
        return scores.most_common(1)[0][0]

    return "Other"

def main():
    print("Loading checkpoint file...")
    data_by_specialty = defaultdict(list)
    other_terms_counter = Counter()

    # Classify
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            entry = json.loads(line)
            specialty = get_specialty(entry.get('mesh_terms', []))

            entry['specialty'] = specialty
            
            data_by_specialty[specialty].append(entry)
            
            # If 'Other', track why (for debugging)
            if specialty == "Other":
                for term in entry.get('mesh_terms', []):
                    other_terms_counter[term] += 1

    # Report Distribution
    print("\n--- Distribution Found ---")
    for spec, items in data_by_specialty.items():
        print(f"{spec}: {len(items)}")

    # The "Fix Your Dictionary" Helper
    if len(data_by_specialty["Other"]) > 0:
        print("\n--- Top Unclassified Terms in 'Other' ---")
        print("(Add these keywords to SPECIALTY_RULES if they are relevant)")
        for term, count in other_terms_counter.most_common(20):
            print(f"  - {term}: {count}")

    # Stratified Sampling
    final_fewshot = []
    final_questions = []
    
    print("\n--- Sampling ---")
    for spec, items in data_by_specialty.items():
        if spec == "Other": continue # Skip 'Other'
        
        if len(items) < TARGET_PER_CLASS:
            print(f"Warning: {spec} only has {len(items)} samples (Target: {TARGET_PER_CLASS}). Taking all.")
            selected = items
        else:
            selected = random.sample(items, TARGET_PER_CLASS)
            
        # Split 50/50
        mid = len(selected) // 2
        final_fewshot.extend(selected[:mid])
        final_questions.extend(selected[mid:])
        print(f"Added {len(selected)} entries from {spec}")

    # Save
    print(f"\nSaving {len(final_fewshot)} few-shots and {len(final_questions)} questions...")
    with open(OUTPUT_FEWSHOT, 'w') as f:
        for item in final_fewshot: f.write(json.dumps(item) + "\n")
        
    with open(OUTPUT_QUESTION, 'w') as f:
        for item in final_questions: f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
