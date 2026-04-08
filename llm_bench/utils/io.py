import json
from pathlib import Path
from typing import List
from llm_bench.schemas.benchmark_result import BenchmarkResult


def save_results_jsonl(results: List[BenchmarkResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")


def load_results_jsonl(path: str | Path) -> List[BenchmarkResult]:
    path = Path(path)
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(BenchmarkResult.model_validate_json(line))
    return results