import argparse
import json
from pathlib import Path
from typing import Optional


def _to_title_label(key: str) -> str:
    """Convert JSON key names into readable Title Case labels."""
    return key.replace("_", " ").upper()


def visualize_dataset(
    input_path: str,
    limit: Optional[int] = None,
    start_idx: int = 0,
    exclude_params: Optional[list[str]] = None,
) -> None:
    """
    Read and display JSONL dataset entries in console for debugging.
    
    Args:
        input_path: Path to JSONL file
        limit: Maximum number of records to display
        start_idx: Starting record index
        exclude_params: List of field names to skip while printing
    """
    path = Path(input_path)
    excluded = set(exclude_params or [])
    
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
                
                print("\n\n" + "="*80)
                print(f"Record #{idx + 1}")
                print("="*80)

                printed_any = False
                for key, value in record.items():
                    if key in excluded:
                        continue

                    printed_any = True
                    print(f"\n{_to_title_label(key)}:")
                    print("-" * 40)
                    if isinstance(value, (dict, list)):
                        print(json.dumps(value, indent=2, ensure_ascii=False))
                    else:
                        print(value)

                if not printed_any:
                    print("\n(No fields to display after exclusions)")
                
        print(f"\n{'='*80}")
        print(f"Displayed {count} record(s)")
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize JSONL dataset entries in console for debugging"
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
    parser.add_argument(
        "-e", "--exclude",
        nargs="*",
        default=[],
        help="Field names to exclude from output (space-separated)"
    )
    
    args = parser.parse_args()
    visualize_dataset(
        args.input_path,
        limit=args.limit,
        start_idx=args.start,
        exclude_params=args.exclude,
    )
