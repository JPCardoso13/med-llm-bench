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


def _average_ranks(values: Sequence[float]) -> list[float]:
    """1-indexed ranks, tied values receiving the average of the ranks they span."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman's rank correlation - Pearson correlation computed on ranks
    instead of raw values, with average-rank tie handling (ties are expected
    here: judge labels are a small ordinal set, not continuous data).
    Dependency-free (no scipy) to match this module's existing style.
    Returns None if there are fewer than 2 pairs or either side is constant
    (undefined correlation, not zero).
    """
    if len(x) != len(y) or len(x) < 2:
        return None

    rank_x = _average_ranks(list(x))
    rank_y = _average_ranks(list(y))
    mean_x = mean(rank_x)
    mean_y = mean(rank_y)

    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    variance_x = sum((a - mean_x) ** 2 for a in rank_x)
    variance_y = sum((b - mean_y) ** 2 for b in rank_y)
    if variance_x == 0 or variance_y == 0:
        return None

    return covariance / (variance_x ** 0.5 * variance_y ** 0.5)