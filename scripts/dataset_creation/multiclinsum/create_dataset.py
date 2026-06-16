"""
Prepares the MultiClinSum (English test set) for benchmarking.

Reads paired fulltext/summary .txt files, sorts by numeric file ID,
takes the first N_FEWSHOT entries as the fewshot pool and the next
N_EVAL entries as the eval split, and writes both to JSONL.

Length bins (computed from source word count):
  short  : < 300 words
  medium : 300-600 words
  long   : > 600 words
"""

import json
from pathlib import Path

RAW_DIR = Path("data/raw/multiclinsum_test_en/multiclinsum_test_en")
FULLTEXT_DIR = RAW_DIR / "fulltext"
SUMMARIES_DIR = RAW_DIR / "summaries"
OUT_DIR = Path("data/processed/multiclinsum")

N_FEWSHOT = 10
N_EVAL = 1000


def _extract_numeric_id(path: Path) -> int:
    # "multiclinsum_test_42_en" -> 42
    stem = path.stem.removeprefix("multiclinsum_test_").removesuffix("_en")
    return int(stem)


def _length_bin(text: str) -> str:
    words = len(text.split())
    if words < 300:
        return "short"
    if words < 600:
        return "medium"
    return "long"


def _build_record(fulltext_path: Path) -> dict:
    file_id = _extract_numeric_id(fulltext_path)
    summary_path = SUMMARIES_DIR / f"multiclinsum_test_{file_id}_en_sum.txt"

    question = fulltext_path.read_text(encoding="utf-8").strip()
    answer = summary_path.read_text(encoding="utf-8").strip()

    return {
        "id": str(file_id),
        "question": question,
        "answer": answer,
        "length_bin": _length_bin(question),
        "file_id": f"multiclinsum_test_{file_id}_en",
    }


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    all_files = sorted(FULLTEXT_DIR.glob("multiclinsum_test_*_en.txt"), key=_extract_numeric_id)

    fewshot_files = all_files[:N_FEWSHOT]
    eval_files = all_files[N_FEWSHOT : N_FEWSHOT + N_EVAL]

    fewshot_records = [_build_record(p) for p in fewshot_files]
    eval_records = [_build_record(p) for p in eval_files]

    _write_jsonl(fewshot_records, OUT_DIR / "fewshot.jsonl")
    _write_jsonl(eval_records, OUT_DIR / "eval.jsonl")

    bin_counts = {"short": 0, "medium": 0, "long": 0}
    for r in eval_records:
        bin_counts[r["length_bin"]] += 1

    print(f"Fewshot : {len(fewshot_records)} entries (IDs {fewshot_records[0]['id']}–{fewshot_records[-1]['id']})")
    print(f"Eval    : {len(eval_records)} entries (IDs {eval_records[0]['id']}–{eval_records[-1]['id']})")
    print(f"Length bins (eval): {bin_counts}")


if __name__ == "__main__":
    main()
