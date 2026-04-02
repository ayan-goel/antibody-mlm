"""Statistical significance utilities for evaluation metrics.

Provides bootstrap confidence intervals for zero-shot metrics (which lack
multi-seed runs) and paired bootstrap tests for comparing two strategies.
Downstream tasks already report multi-seed std via _aggregate_seeds().
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    values: np.ndarray | list[float],
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Args:
        values: Array of observed values.
        statistic_fn: Function that computes the statistic from a sample.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.RandomState(seed)

    point_estimate = float(statistic_fn(values))
    bootstrap_stats = np.empty(n_bootstrap)
    n = len(values)

    for i in range(n_bootstrap):
        sample = values[rng.randint(0, n, size=n)]
        bootstrap_stats[i] = statistic_fn(sample)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper


def paired_bootstrap_test(
    values_a: np.ndarray | list[float],
    values_b: np.ndarray | list[float],
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided paired bootstrap test: is statistic(A) != statistic(B)?

    Tests whether the difference in a statistic between two paired samples
    is significantly different from zero.

    Args:
        values_a: Observations from system A.
        values_b: Observations from system B (same indices as A).
        statistic_fn: Function that computes the statistic from a sample.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Two-sided p-value (fraction of resamples where the sign of the
        difference flips relative to the observed difference).
    """
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    assert len(values_a) == len(values_b), "Paired samples must have equal length"

    rng = np.random.RandomState(seed)
    n = len(values_a)

    observed_diff = statistic_fn(values_a) - statistic_fn(values_b)
    count = 0

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        diff = statistic_fn(values_a[idx]) - statistic_fn(values_b[idx])
        if observed_diff >= 0 and diff < 0:
            count += 1
        elif observed_diff < 0 and diff >= 0:
            count += 1

    return count / n_bootstrap


def format_ci(
    point: float, lower: float, upper: float, fmt: str = ".4f"
) -> str:
    """Format a point estimate with CI as 'point [lower, upper]'."""
    return f"{point:{fmt}} [{lower:{fmt}}, {upper:{fmt}}]"
