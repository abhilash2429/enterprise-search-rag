"""Near-duplicate detection by embedding similarity: doc vectors pooled from chunk vectors, exact kNN on GPU,
pairs kept with their cosine so the clustering threshold is chosen later from labeled dev pairs."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from entsearch.data import ROOT

DENSE = ROOT / "data/index/dense_qwen3-0.6b_c512"
OUT = ROOT / "data/index/neardup_embed"


def doc_vectors(dense_dir: Path = DENSE, pooling: str = "mean") -> tuple[list[str], np.ndarray]:
    """One L2-normalized vector per doc. Chunks of a doc are contiguous within a shard."""
    ids, vecs = [], []
    for npy in sorted(dense_dir.glob("shard_*.npy")):
        meta = pq.read_table(npy.with_suffix(".parquet")).to_pandas()
        v = np.load(npy).astype(np.float32)
        starts = np.flatnonzero(meta.chunk_idx.to_numpy() == 0)
        assert (meta.doc_id.to_numpy()[starts[1:]] != meta.doc_id.to_numpy()[starts[:-1]]).all()
        if pooling == "mean":
            pooled = np.add.reduceat(v, starts, axis=0)
        elif pooling == "first":
            pooled = v[starts]
        else:
            raise ValueError(pooling)
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
        ids.extend(meta.doc_id.to_numpy()[starts])
        vecs.append(pooled.astype(np.float16))
    return ids, np.concatenate(vecs)


def knn_pairs(vecs: np.ndarray, k: int = 10, floor: float = 0.8, block: int = 1024) -> np.ndarray:
    """(i, j, cos) for each doc's top-k neighbours with cos >= floor, i < j, deduplicated."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torch.from_numpy(vecs).to(dev)
    if dev == "cpu":
        m = m.float()
    out_i, out_j, out_s = [], [], []
    for start in tqdm(range(0, len(m), block), desc="knn"):
        s = m[start : start + block] @ m.T
        rows = torch.arange(start, min(start + block, len(m)), device=dev)
        s[torch.arange(len(rows), device=dev), rows] = -1  # drop self-match
        val, idx = torch.topk(s.float(), k, dim=1)
        keep = val >= floor
        out_i.append(rows.unsqueeze(1).expand_as(idx)[keep].cpu().numpy())
        out_j.append(idx[keep].cpu().numpy())
        out_s.append(val[keep].cpu().numpy())
    i, j, s = np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_s)
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    _, first = np.unique(lo.astype(np.int64) * len(vecs) + hi, return_index=True)
    pairs = np.zeros(len(first), dtype=[("i", np.int32), ("j", np.int32), ("cos", np.float32)])
    pairs["i"], pairs["j"], pairs["cos"] = lo[first], hi[first], s[first]
    return pairs


def pair_cosines(ids: list[str], vecs: np.ndarray, pairs: list[tuple[str, str]]) -> np.ndarray:
    pos = {d: n for n, d in enumerate(ids)}
    return np.array([float(vecs[pos[a]].astype(np.float32) @ vecs[pos[b]].astype(np.float32)) for a, b in pairs])
