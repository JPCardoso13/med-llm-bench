import argparse
import json
from pathlib import Path
from typing import Optional


def visualize_dataset(input_path: str, limit: Optional[int] = None, start_idx: int = 0) -> None:
    """
    Read and display MCQ dataset entries in console for debugging.
    
    Args:
        input_path: Path to JSONL file
        limit: Maximum number of records to display
        start_idx: Starting record index
    """
    path = Path(input_path)
    
    if not path.exists():
        print(f"Error: File not found: {input_path}")
        return
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            count = 0
            for idx, line in enumerate(f):
                if idx < start_idx:
                    continue
                    
                if limit and count >= limit:
                    break
                
                count += 1
                record = json.loads(line)
                
                print("\n" + "="*80)
                print(f"Record #{idx + 1}")
                print("="*80)
                
                print(f"\nCase ID: {record.get('pmc_id', 'N/A')}")
                print(f"Article Link: {record.get('article_link', 'N/A')}")
                
                print(f"\nCase Prompt:")
                print("-" * 40)
                print(record.get('case_prompt', 'N/A'))
                
                print(f"\nDiagnostic Reasoning:")
                print("-" * 40)
                print(record.get('diagnostic_reasoning', 'N/A'))
                
                print(f"\nFinal Diagnosis:")
                print("-" * 40)
                print(record.get('final_diagnosis', 'N/A'))
                
                print(f"\nDistractors:")
                print("-" * 40)
                distractors = record.get('distractors', [])
                for i, distractor in enumerate(distractors, 1):
                    print(f"  {i}. {distractor}")
                
        print(f"\n{'='*80}")
        print(f"Displayed {count} record(s)")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize MCQ dataset entries in console for debugging"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to JSONL file"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=None,
        help="Maximum number of records to display"
    )
    parser.add_argument(
        "-s", "--start",
        type=int,
        default=0,
        help="Starting record index (0-based)"
    )
    
    args = parser.parse_args()
    visualize_dataset(args.input_path, limit=args.limit, start_idx=args.start)
