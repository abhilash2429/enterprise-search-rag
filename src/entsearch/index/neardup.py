"""Near-duplicate detection: MinHash over word shingles, LSH candidates, union-find clusters.

Pairs are kept with their estimated Jaccard down to a low floor, so the clustering threshold can be chosen
later from the measured distribution without recomputing signatures.
"""

import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from entsearch.analysis import tokenize
from entsearch.data import ROOT, doc_text, iter_corpus

OUT = ROOT / "data/index/neardup"
NUM_PERM = 128
SHINGLE = 5
SEED = 1
SCHEME = "affine32"  # datasketch 2.x default; must be passed when rebuilding from stored hashvalues


def shingles(text: str, k: int = SHINGLE) -> set[bytes]:
    toks = tokenize(text)
    if len(toks) < k:
        return {" ".join(toks).encode()} if toks else set()
    return {" ".join(toks[i : i + k]).encode() for i in range(len(toks) - k + 1)}


def signature(text: str) -> np.ndarray:
    m = MinHash(num_perm=NUM_PERM, seed=SEED, scheme=SCHEME)
    m.update_batch(list(shingles(text)))
    return m.hashvalues


def _sign_batch(texts: list[str]) -> np.ndarray:
    return np.stack([signature(t) for t in texts])


def build_signatures(out: Path = OUT, workers: int = 6, batch_size: int = 2_000) -> None:
    # Process-parallel work: one BLAS thread per worker, or N workers x N threads exhaust memory on startup.
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    doc_ids, sigs = [], []

    def batches():
        for batch in iter_corpus(batch_size=batch_size, columns=("doc_id", "title", "content")):
            doc_ids.extend(r["doc_id"] for r in batch)
            yield [doc_text(r["title"], r["content"]) for r in batch]

    with ProcessPoolExecutor(workers) as ex, tqdm(desc="minhash", unit="doc") as bar:
        pending = deque()
        for texts in batches():
            pending.append(ex.submit(_sign_batch, texts))
            if len(pending) >= 2 * workers:
                s = pending.popleft().result()
                sigs.append(s)
                bar.update(len(s))
        while pending:
            s = pending.popleft().result()
            sigs.append(s)
            bar.update(len(s))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "signatures.npy", np.concatenate(sigs))
    (out / "doc_ids.txt").write_text("\n".join(doc_ids) + "\n", encoding="utf8")


def candidate_pairs(sigs: np.ndarray, floor: float) -> np.ndarray:
    """(i, j, est_jaccard) for i < j with estimated Jaccard >= floor, as a structured array."""
    lsh = MinHashLSH(threshold=floor, num_perm=sigs.shape[1])
    for i, hv in enumerate(tqdm(sigs, desc="lsh insert")):
        lsh.insert(i, MinHash(num_perm=sigs.shape[1], seed=SEED, hashvalues=hv, scheme=SCHEME), check_duplication=False)
    out_i, out_j, out_s = [], [], []
    for i, hv in enumerate(tqdm(sigs, desc="lsh query")):
        cand = [j for j in lsh.query(MinHash(num_perm=sigs.shape[1], seed=SEED, hashvalues=hv, scheme=SCHEME)) if j > i]
        if not cand:
            continue
        cand = np.asarray(cand)
        est = (sigs[cand] == hv).mean(axis=1)
        keep = est >= floor
        out_i.append(np.full(keep.sum(), i, dtype=np.int32))
        out_j.append(cand[keep].astype(np.int32))
        out_s.append(est[keep].astype(np.float32))
    pairs = np.zeros(sum(len(x) for x in out_i), dtype=[("i", np.int32), ("j", np.int32), ("jaccard", np.float32)])
    if len(pairs):
        pairs["i"], pairs["j"], pairs["jaccard"] = np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_s)
    return pairs


def clusters(n: int, pairs: np.ndarray, threshold: float) -> np.ndarray:
    """Union-find over pairs with jaccard >= threshold. Returns a cluster id per doc (the root's index)."""
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs[pairs["jaccard"] >= threshold][["i", "j"]]:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(x) for x in range(n)])
