"""Per-question retrieval metrics. Tests: tests/test_metrics.py.

Shared contract:
  retrieved  ranked doc ids, best first, no duplicates (callers dedupe, like the harness does)
  relevant   gold doc ids, non-empty (questions without gold are excluded upstream; raise ValueError if empty)
  k          cutoff; only retrieved[:k] counts
Binary relevance throughout.
"""

import math
from collections.abc import Collection, Sequence


def _gold(relevant: Collection[str], k: int) -> set[str]:
    if not relevant:
        raise ValueError("relevant must be non-empty")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return set(relevant)


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """|top-k ∩ relevant| / |relevant|. Matches the harness's document_recall_pct / 100 at k = 10."""
    gold = _gold(relevant, k)
    return sum(d in gold for d in retrieved[:k]) / len(gold)


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 / rank of the first relevant doc within top-k (rank starts at 1), else 0."""
    gold = _gold(relevant, k)
    return next((1 / r for r, d in enumerate(retrieved[:k], 1) if d in gold), 0.0)


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """DCG = sum over hits at rank r (1-based) of 1 / log2(r + 1). IDCG puts min(|relevant|, k) hits at the top."""
    gold = _gold(relevant, k)
    dcg = sum(1 / math.log2(r + 1) for r, d in enumerate(retrieved[:k], 1) if d in gold)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / idcg
