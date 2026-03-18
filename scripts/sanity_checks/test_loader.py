import argparse
import logging
import json
from pprint import pprint

# Adjust these imports based on your actual directory structure
from llm_bench.ingestion import YamlLoader
from llm_bench.schemas import MCQSample, GenerativeSample

# Set up logging to see the loader's output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def test_dataset_config(yaml_path: str):
    print(f"\n{'='*50}")
    print(f"Testing Config: {yaml_path}")
    print(f"{'='*50}\n")

    try:
        # 1. Instantiate the loader with the YAML config
        loader = YamlLoader(yaml_path)
        
        # 2. Run the load method
        splits = loader.load()
        
        eval_samples = splits.get("eval", [])
        fewshot_samples = splits.get("fewshot", [])

        # 3. Print high-level stats
        print("\n--- Load Statistics ---")
        print(f"Eval Samples Loaded:    {len(eval_samples)}")
        print(f"Fewshot Samples Loaded: {len(fewshot_samples)}")

        # 4. Deep dive into the first Eval sample to verify Pydantic validation
        if eval_samples:
            print("\n--- Inspecting First Eval Sample ---")
            first_sample = eval_samples[0]
            
            # Verify the type
            print(f"Schema Type: {type(first_sample).__name__}")
            
            # Print the validated Pydantic object as a formatted dictionary
            # Using model_dump() (Pydantic V2) or dict() (Pydantic V1)
            dumped_model = first_sample.model_dump() if hasattr(first_sample, 'model_dump') else first_sample.dict()
            pprint(dumped_model, sort_dicts=False)
            
            # Specifically check if the smart coercion worked
            grouping = dumped_model.get("grouping", {})
            print("\n--- Grouping Check ---")
            for key, val in grouping.items():
                print(f"Key '{key}' is type: {type(val).__name__} -> {val}")
                if not isinstance(val, list):
                    print("WARNING: Grouping value is not a list!")

    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a dataset YAML configuration.")
    parser.add_argument("config_path", help="Path to the YAML config file")
    args = parser.parse_args()

    test_dataset_config(args.config_path)