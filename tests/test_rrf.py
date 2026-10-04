import pytest

from entsearch.retrieval.rrf import rrf


def test_scores_and_order():
    fused = rrf([["a", "b"], ["b", "c"]], k=60)
    assert [d for d, _ in fused] == ["b", "a", "c"]
    assert dict(fused) == pytest.approx({"b": 1 / 61 + 1 / 62, "a": 1 / 61, "c": 1 / 62})


def test_top_n_and_empty():
    assert [d for d, _ in rrf([["a", "b", "c"]], top_n=2)] == ["a", "b"]
    assert rrf([]) == []
    assert rrf([[], []]) == []


def test_duplicate_within_ranking_counts_once_at_best_rank():
    assert dict(rrf([["a", "b", "a"]], k=0)) == pytest.approx({"a": 1.0, "b": 0.5})


def test_ties_keep_first_appearance():
    assert [d for d, _ in rrf([["x", "y"], ["y", "x"]])] == ["x", "y"]
    assert [d for d, _ in rrf([["p"], ["q"]])] == ["p", "q"]


def test_negative_k_raises():
    with pytest.raises(ValueError):
        rrf([["a"]], k=-1)
