from .generativesample import GenerativeSample
from .mcqsample import MCQSample

__all__ = ["MCQSample", "GenerativeSample"]

SCHEMA_REGISTRY = {
    "GenerativeSample": GenerativeSample,
    "MCQSample": MCQSample,
}