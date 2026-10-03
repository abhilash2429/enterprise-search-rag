"""Per-question retrieval metrics. HAND-WRITTEN BY ABHILASH. Tests: tests/test_metrics.py.

Shared contract:
  retrieved  ranked doc ids, best first, no duplicates (callers dedupe, like the harness does)
  relevant   gold doc ids, non-empty (questions without gold are excluded upstream; raise ValueError if empty)
  k          cutoff; only retrieved[:k] counts
Binary relevance throughout.
"""

from collections.abc import Collection, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """|top-k ∩ relevant| / |relevant|. Matches the harness's document_recall_pct / 100 at k = 10."""
    raise NotImplementedError


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 / rank of the first relevant doc within top-k (rank starts at 1), else 0."""
    raise NotImplementedError


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """DCG = sum over hits at rank r (1-based) of 1 / log2(r + 1). IDCG puts min(|relevant|, k) hits at the top."""
    raise NotImplementedError
