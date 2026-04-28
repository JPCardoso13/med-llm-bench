import json
from pathlib import Path
from typing import List
from llm_bench.schemas.benchmark_result import BenchmarkResult


def save_results_jsonl(results: List[BenchmarkResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")


def save_results_json(results: List[BenchmarkResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump(mode="json") for result in results]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_results(results: List[BenchmarkResult], path: str | Path) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        save_results_json(results, path)
        return
    if suffix == ".jsonl":
        save_results_jsonl(results, path)
        return
    raise ValueError(f"Unsupported results extension: {suffix}. Use .json or .jsonl")


def load_results_jsonl(path: str | Path) -> List[BenchmarkResult]:
    path = Path(path)
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(BenchmarkResult.model_validate_json(line))
    return results