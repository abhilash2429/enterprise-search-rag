import numpy as np
import pytest

from entsearch.stats import bootstrap_ci, paired_bootstrap


def test_constant_scores_give_degenerate_interval():
    assert bootstrap_ci([0.7] * 50) == pytest.approx((0.7, 0.7, 0.7))


def test_interval_brackets_mean_and_matches_normal_approx():
    x = np.random.default_rng(1).binomial(1, 0.6, 500).astype(float)
    mean, lo, hi = bootstrap_ci(x)
    assert lo < mean < hi
    se = x.std() / np.sqrt(len(x))
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.1)


def test_deterministic_per_seed():
    x = np.random.default_rng(2).random(100)
    assert bootstrap_ci(x, seed=3) == bootstrap_ci(x, seed=3)
    assert bootstrap_ci(x, seed=3) != bootstrap_ci(x, seed=4)


def test_paired_identical_systems_is_zero():
    x = np.random.default_rng(3).random(200)
    assert paired_bootstrap(x, x) == (0.0, 0.0, 0.0)


def test_paired_detects_consistent_gain_that_unpaired_would_miss():
    rng = np.random.default_rng(4)
    base = rng.random(300)
    system = base + 0.02  # tiny gain, huge between-question variance
    delta, lo, hi = paired_bootstrap(system, base)
    assert delta == pytest.approx(0.02) and lo > 0
    _, base_lo, base_hi = bootstrap_ci(base)
    assert base_hi - base_lo > 0.02  # unpaired intervals would overlap


def test_input_validation():
    with pytest.raises(ValueError):
        bootstrap_ci([])
    with pytest.raises(ValueError):
        bootstrap_ci([0.1, np.nan])
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])
