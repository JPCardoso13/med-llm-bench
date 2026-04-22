from __future__ import annotations

from statistics import mean, pstdev
from typing import Iterable, Sequence


def as_float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def percentile(values: Sequence[float] | Iterable[float], p: float) -> float | None:
    numbers = sorted(as_float_list(values))
    if not numbers:
        return None

    if p <= 0:
        return numbers[0]
    if p >= 100:
        return numbers[-1]

    position = (len(numbers) - 1) * (p / 100.0)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(numbers) - 1)
    fraction = position - lower_index

    if lower_index == upper_index:
        return numbers[lower_index]

    lower_value = numbers[lower_index]
    upper_value = numbers[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def aggregate_values(values: Iterable[float], aggregates: Sequence[str], percentiles: Sequence[int] | None = None) -> dict[str, float | None]:
    numbers = as_float_list(values)
    summary: dict[str, float | None] = {}

    for aggregate in aggregates:
        if aggregate == "mean":
            summary["mean"] = mean(numbers) if numbers else None
        elif aggregate == "std":
            if not numbers:
                summary["std"] = None
            elif len(numbers) == 1:
                summary["std"] = 0.0
            else:
                summary["std"] = pstdev(numbers)
        elif aggregate == "min":
            summary["min"] = min(numbers) if numbers else None
        elif aggregate == "max":
            summary["max"] = max(numbers) if numbers else None
        elif aggregate == "sum":
            summary["sum"] = sum(numbers) if numbers else None
        else:
            raise ValueError(f"Unsupported aggregate: {aggregate}")

    for pct in percentiles or []:
        summary[f"p{pct}"] = percentile(numbers, float(pct))

    return summary