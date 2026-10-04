"""Reciprocal rank fusion (Cormack et al. 2009). Tests: tests/test_rrf.py.

  score(d) = sum over rankings containing d of 1 / (k + rank(d)),  rank 1-based
Only ranks are used, so lists with incomparable score scales (BM25, cosine) fuse without normalization.
"""

from collections.abc import Hashable, Sequence


def rrf(rankings: Sequence[Sequence[Hashable]], k: int = 60, top_n: int | None = None) -> list[tuple[Hashable, float]]:
    """Fused (id, score) pairs, best first, at most top_n.

    An id repeated within one ranking counts once, at its best rank. Ties keep the order in which ids first appear
    across the rankings (first ranking first), so output is deterministic.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        seen = set()
        for rank, d in enumerate(ranking, 1):
            if d in seen:
                continue
            seen.add(d)
            scores[d] = scores.get(d, 0.0) + 1 / (k + rank)
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return fused if top_n is None else fused[:top_n]
