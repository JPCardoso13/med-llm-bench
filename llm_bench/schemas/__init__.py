from .generativesample import GenerativeSample
from .mcqsample import MCQSample
from .benchmark_result import BenchmarkResult

__all__ = ["MCQSample", "GenerativeSample", "BenchmarkResult"]

SCHEMA_REGISTRY = {
    "GenerativeSample": GenerativeSample,
    "MCQSample": MCQSample,
    "BenchmarkResult": BenchmarkResult
}
