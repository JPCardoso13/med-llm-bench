"""Remove metamap_phrases from MEDQA JSONL files."""

import json
from pathlib import Path
from typing import Iterable, Dict, Any


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def remove_key(records: Iterable[Dict[str, Any]], key: str) -> Iterable[Dict[str, Any]]:
    for record in records:
        if key in record:
            record = {k: v for k, v in record.items() if k != key}
        yield record


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "data" / "raw" / "medqa"
    output_dir = base_dir / "data" / "processed" / "medqa"
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (input_dir / "dev.jsonl", output_dir / "fewshots.jsonl"),
        (input_dir / "test.jsonl", output_dir / "questions.jsonl"),
    ]

    for input_path, output_path in jobs:
        records = iter_jsonl(input_path)
        cleaned = remove_key(records, "metamap_phrases")
        write_jsonl(output_path, cleaned)


if __name__ == "__main__":
    main()
