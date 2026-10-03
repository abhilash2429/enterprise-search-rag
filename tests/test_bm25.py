import time

import bm25s
import numpy as np
import pytest
import scipy.sparse as sp

from entsearch.retrieval.bm25 import BM25


def random_corpus(rng, n_docs=300, n_terms=80):
    docs = []
    for _ in range(n_docs):
        length = int(rng.integers(1, 60))
        docs.append(list(rng.zipf(1.3, size=length) % n_terms))
    present = sorted({t for d in docs for t in d})
    remap = {t: i for i, t in enumerate(present)}
    docs = [[remap[t] for t in d] for d in docs]
    return docs, len(present)


def to_tf(docs, n_terms):
    rows = [i for i, d in enumerate(docs) for _ in d]
    cols = [t for d in docs for t in d]
    tf = sp.coo_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(len(docs), n_terms)).tocsc()
    tf.sum_duplicates()
    return tf, np.array([len(d) for d in docs], dtype=np.int32)


def oracle(docs, k1, b):
    ref = bm25s.BM25(method="lucene", k1=k1, b=b)
    ref.index([[str(t) for t in d] for d in docs], show_progress=False)
    return lambda q: ref.get_scores([str(t) for t in q])


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (0.9, 0.4), (2.0, 1.0), (1.2, 0.0)])
def test_scores_match_lucene(seed, k1, b):
    rng = np.random.default_rng(seed)
    docs, n_terms = random_corpus(rng)
    tf, doc_len = to_tf(docs, n_terms)
    model = BM25(k1, b).fit(tf, doc_len)
    ref = oracle(docs, k1, b)
    for _ in range(20):
        q = list(rng.integers(0, n_terms, size=int(rng.integers(1, 8))))
        np.testing.assert_allclose(model.score(q), ref(q), rtol=1e-5, atol=1e-6)


def test_repeated_query_term_counts_per_occurrence():
    docs, n_terms = random_corpus(np.random.default_rng(0))
    model = BM25().fit(*to_tf(docs, n_terms))
    np.testing.assert_allclose(model.score([3, 3]), 2 * model.score([3]), rtol=1e-6)


def test_no_match_scores_zero_and_empty_query():
    docs = [[0, 1], [1, 2], [3]]
    model = BM25().fit(*to_tf(docs, 4))
    s = model.score([0])
    assert s[0] > 0 and s[1] == 0 and s[2] == 0
    assert np.all(model.score([]) == 0)
    idx, scores = model.topk([], k=5)
    assert len(idx) == 0 and len(scores) == 0


def test_topk_order_ties_and_positive_only():
    docs = [[5], [0, 1], [0, 1], [0], [2, 2, 2], [0, 0, 0, 0]]
    model = BM25().fit(*to_tf(docs, 6))
    full = model.score([0])
    idx, scores = model.topk([0], k=10)
    assert all(full[i] > 0 for i in idx)
    assert len(idx) == int((full > 0).sum())
    assert list(scores) == sorted(scores, reverse=True)
    np.testing.assert_allclose(scores, full[idx])
    assert list(idx).index(1) < list(idx).index(2)  # docs 1 and 2 tie; lower index first
    idx2, _ = model.topk([0], k=2)
    assert list(idx2) == list(idx[:2])


def test_scales_to_corpus_size():
    rng = np.random.default_rng(0)
    n_docs, n_terms, nnz = 512_000, 200_000, 20_000_000
    rows = rng.integers(0, n_docs, nnz, dtype=np.int32)
    cols = (rng.zipf(1.2, nnz) % n_terms).astype(np.int32)
    tf = sp.csc_matrix((np.ones(nnz, dtype=np.int32), (rows, cols)), shape=(n_docs, n_terms))
    model = BM25().fit(tf, np.asarray(tf.sum(axis=1)).ravel().astype(np.int32))
    q = [int(t) for t in rng.integers(0, 2_000, 15)]
    model.topk(q, k=10)
    t0 = time.perf_counter()
    for _ in range(10):
        model.topk(q, k=10)
    assert (time.perf_counter() - t0) / 10 < 0.25
