import pytest

from entsearch.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall():
    r = ["a", "b", "c", "d"]
    assert recall_at_k(r, {"b", "d", "x"}, 2) == pytest.approx(1 / 3)
    assert recall_at_k(r, {"b", "d", "x"}, 4) == pytest.approx(2 / 3)
    assert recall_at_k(r, ["b", "d", "x"], 10) == pytest.approx(2 / 3)
    assert recall_at_k(r, {"z"}, 4) == 0.0
    assert recall_at_k([], {"z"}, 10) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"c"}, 3) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], {"c"}, 2) == 0.0
    assert reciprocal_rank(["a", "b", "c"], {"b", "a"}, 3) == 1.0


def test_ndcg():
    assert ndcg_at_k(["a", "b", "c"], {"b", "c"}, 3) == pytest.approx(0.693426, abs=1e-6)
    assert ndcg_at_k(["a", "b"], {"a", "b", "c"}, 2) == 1.0
    assert ndcg_at_k(["x", "a"], {"a"}, 2) == pytest.approx(0.630930, abs=1e-6)
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert ndcg_at_k(["x", "a"], {"a"}, 1) == 0.0


@pytest.mark.parametrize("fn", [recall_at_k, reciprocal_rank, ndcg_at_k])
def test_empty_relevant_raises(fn):
    with pytest.raises(ValueError):
        fn(["a"], set(), 10)
