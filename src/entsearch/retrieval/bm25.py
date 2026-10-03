"""BM25 scorer. HAND-WRITTEN BY ABHILASH. Tests: tests/test_bm25.py.

Target: Lucene BM25Similarity (what OpenSearch uses), so recall matches the paper baseline.
  idf(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
  score(d, q) = sum over query terms t (repeats count once per occurrence) of
                idf(t) * tf(t,d) / (tf(t,d) + k1 * (1 - b + b * len(d) / avgdl))
  avgdl       = mean doc length over the corpus
Lucene dropped the (k1 + 1) numerator factor in v8. It is rank-neutral, but the tests expect it absent.

Inputs come from entsearch.index.sparse.SparseIndex:
  tf       scipy.sparse.csc_matrix, shape (n_docs, n_terms), raw int counts (column = term)
  doc_len  np.ndarray[int], shape (n_docs,), tokens per doc
Full corpus is ~512K docs, so score() must be sparse/vectorized: one call should take milliseconds, not seconds.
"""

import numpy as np
import scipy.sparse as sp


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, tf: sp.csc_matrix, doc_len: np.ndarray) -> "BM25":
        """Precompute whatever score() needs. Returns self."""
        raise NotImplementedError

    def score(self, query_terms: list[int]) -> np.ndarray:
        """Score of every doc for the query, shape (n_docs,), float. Docs matching no term score 0."""
        raise NotImplementedError

    def topk(self, query_terms: list[int], k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """(doc_indices, scores) of the k best docs, best first. Only docs with score > 0. Ties: lower doc index first."""
        raise NotImplementedError
