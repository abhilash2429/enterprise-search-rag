import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from entsearch.index.neardup import clusters
from entsearch.index.neardup_embed import doc_vectors, knn_pairs


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_doc_vectors_pool_contiguous_chunks(tmp_path):
    rng = np.random.default_rng(0)
    v = _unit(rng.normal(size=(5, 8))).astype(np.float16)
    np.save(tmp_path / "shard_000.npy", v)
    pq.write_table(pa.table({"doc_id": ["a", "a", "b", "c", "c"], "chunk_idx": [0, 1, 0, 0, 1]}), tmp_path / "shard_000.parquet")
    ids, mean = doc_vectors(tmp_path, "mean")
    _, first = doc_vectors(tmp_path, "first")
    assert ids == ["a", "b", "c"]
    np.testing.assert_allclose(mean[0], _unit(v[0].astype(np.float32) + v[1].astype(np.float32)), atol=2e-3)
    np.testing.assert_allclose(first[2], v[3], atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(mean.astype(np.float32), axis=1), 1, atol=2e-3)


def test_knn_pairs_find_near_copies_only():
    rng = np.random.default_rng(1)
    base = _unit(rng.normal(size=(50, 64)))
    near = _unit(base[:5] + 0.02 * rng.normal(size=(5, 64)))
    vecs = np.concatenate([base, near]).astype(np.float16)
    pairs = knn_pairs(vecs, k=3, floor=0.9, block=16)
    got = {(int(i), int(j)) for i, j, _ in pairs}
    assert got == {(k, 50 + k) for k in range(5)}
    assert (pairs["i"] < pairs["j"]).all()
    c = clusters(len(vecs), pairs, 0.9, field="cos")
    assert all(c[k] == c[50 + k] for k in range(5)) and len(set(c)) == 50
