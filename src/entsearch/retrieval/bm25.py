"""BM25 scorer. Tests: tests/test_bm25.py.

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

from collections import Counter

import numpy as np
import scipy.sparse as sp

_CHUNK = 1 << 24  # nonzeros per block in fit(), bounds temporary memory on the full corpus


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, tf: sp.csc_matrix, doc_len: np.ndarray) -> "BM25":
        """Precompute whatever score() needs. Returns self.

        Stores, per nonzero (doc, term), the length-normalized saturated tf in float32, laid out like tf's CSC arrays.
        idf stays per term and is applied at query time, so score() is one slice-and-add per query term.
        Assumes canonical CSC (no duplicate or explicit-zero entries), so nonzeros per column = document frequency.
        """
        tf = sp.csc_matrix(tf)
        n_docs, n_terms = tf.shape
        doc_len = np.asarray(doc_len, dtype=np.float64)
        df = np.diff(tf.indptr)
        self.idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        norm = (self.k1 * (1 - self.b + self.b * doc_len / doc_len.mean())).astype(np.float32)
        weights = np.empty(tf.nnz, dtype=np.float32)
        for s in range(0, tf.nnz, _CHUNK):
            v = tf.data[s : s + _CHUNK].astype(np.float32)
            weights[s : s + _CHUNK] = v / (v + norm[tf.indices[s : s + _CHUNK]])
        self.n_docs = n_docs
        self.indptr, self.indices, self.weights = tf.indptr, tf.indices, weights
        return self

    def score(self, query_terms: list[int]) -> np.ndarray:
        """Score of every doc for the query, shape (n_docs,), float. Docs matching no term score 0."""
        out = np.zeros(self.n_docs)
        for t, count in Counter(query_terms).items():
            s, e = self.indptr[t], self.indptr[t + 1]
            out[self.indices[s:e]] += (count * self.idf[t]) * self.weights[s:e]
        return out

    def topk(self, query_terms: list[int], k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """(doc_indices, scores) of the k best docs, best first. Only docs with score > 0. Ties: lower doc index first."""
        scores = self.score(query_terms)
        cand = np.flatnonzero(scores > 0)
        if k <= 0 or len(cand) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0)
        if len(cand) > k:
            # Keep every doc tied with the k-th best score, so the index tie-break below sees all of them.
            kth = np.partition(scores[cand], len(cand) - k)[len(cand) - k]
            cand = cand[scores[cand] >= kth]
        order = np.lexsort((cand, -scores[cand]))[:k]
        idx = cand[order]
        return idx, scores[idx]
