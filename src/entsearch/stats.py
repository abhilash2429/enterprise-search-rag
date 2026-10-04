"""Bootstrap confidence intervals over questions. Tests: tests/test_stats.py.

Inputs are per-question scores (one float per question, e.g. recall@10 or judge correctness).
With several generation seeds, average each question over seeds first, then bootstrap over questions.
Percentile intervals: resample questions with replacement, take the alpha/2 and 1 - alpha/2 quantiles of the mean.
"""

import numpy as np

_BLOCK = 1_000  # resamples per block, bounds memory at _BLOCK x n_questions indices


def bootstrap_ci(scores, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi) of the mean of scores."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError(f"scores must be a non-empty 1-D array, got shape {x.shape}")
    if np.isnan(x).any():
        raise ValueError("scores contain NaN")
    rng = np.random.default_rng(seed)
    means = np.concatenate([
        x[rng.integers(0, len(x), (min(_BLOCK, n_resamples - s), len(x)))].mean(axis=1)
        for s in range(0, n_resamples, _BLOCK)
    ])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(system, baseline, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(delta, lo, hi) for mean(system - baseline) over the same questions in the same order.

    Paired: each resample draws one set of questions and scores both systems on it, which is the same as
    bootstrapping the per-question differences. The gain is real at level alpha when the interval excludes 0.
    """
    a, b = np.asarray(system, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"system and baseline must align question by question, got {a.shape} vs {b.shape}")
    return bootstrap_ci(a - b, n_resamples, alpha, seed)
