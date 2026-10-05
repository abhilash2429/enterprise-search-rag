"""Dense quantization ablation over the Qwen3 c512 chunk index: fp16 exact vs int8 vs binary vs binary + fp16 rescore.

Each variant takes the top-4000 chunks per query, MaxP to the top-100 docs (same as dump_candidates.py dense), and writes
runs/dense_qwen3_<v>/{candidates,retrieval}.jsonl plus runs/hybrid_rrf_k60_<v>/retrieval.jsonl (RRF k=60 with
bm25_own, the headline hybrid). Scores on CPU in row blocks, so it runs while the GPU is busy.

  int8      per-dimension affine scalar quantization, range = [min, max] over all chunks; query stays float
  binary    sign bit per dimension, Hamming distance (ranked via the equivalent +-1 dot product)
  binary_rs binary top-16000 (4x oversample), rescored with the fp16 vectors, then top-4000
"""
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from entsearch.answer import read_jsonl
from entsearch.data import ROOT
from entsearch.retrieval.rrf import rrf

INDEX = ROOT / "data/index/dense_qwen3-0.6b_c512"
CHUNK_LIMIT, OVERSAMPLE, DEPTH, TOP_K, BLOCK = 4000, 4, 100, 10, 200_000

shards = sorted(INDEX.glob("shard_*.npy"))
vecs = np.concatenate([np.load(s) for s in shards])
doc_ids = np.concatenate([pq.read_table(s.with_suffix(".parquet"), columns=["doc_id"])["doc_id"].to_numpy() for s in shards])
assert len(doc_ids) == len(vecs) == 1_538_921 and vecs.dtype == np.float16, (len(doc_ids), vecs.shape, vecs.dtype)
queries = torch.from_numpy(np.load(INDEX / "queries.npy")).float()
qids = json.loads((INDEX / "queries.json").read_text())
n, dim = vecs.shape
torch.set_num_threads(8)

lo = np.full(dim, np.inf, np.float32)
hi = np.full(dim, -np.inf, np.float32)
for s in range(0, n, BLOCK):
    b = vecs[s : s + BLOCK].astype(np.float32)
    lo, hi = np.minimum(lo, b.min(0)), np.maximum(hi, b.max(0))
step = (hi - lo) / 255
codes = np.empty((n, dim), np.uint8)
signs = np.empty((n, dim // 8), np.uint8)
for s in range(0, n, BLOCK):
    b = vecs[s : s + BLOCK].astype(np.float32)
    codes[s : s + BLOCK] = np.clip(np.rint((b - lo) / step), 0, 255).astype(np.uint8)
    signs[s : s + BLOCK] = np.packbits(b > 0, axis=1)
lo_t, step_t = torch.from_numpy(lo), torch.from_numpy(step)
q_sign = (queries > 0).float() * 2 - 1


def block_scores(variant: str, s: int) -> torch.Tensor:
    if variant == "fp16":
        return queries @ torch.from_numpy(vecs[s : s + BLOCK].astype(np.float32)).T
    if variant == "int8":
        x = torch.from_numpy(codes[s : s + BLOCK]).float()
        return (queries * step_t) @ x.T + (queries @ lo_t)[:, None]
    bits = torch.from_numpy(np.unpackbits(signs[s : s + BLOCK], axis=1)).float() * 2 - 1
    return q_sign @ bits.T  # = dim - 2 * Hamming


def search(variant: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k chunk rows and scores per query, merged across row blocks."""
    best_s = torch.full((len(queries), 0), -np.inf)
    best_i = torch.zeros((len(queries), 0), dtype=torch.long)
    for s in range(0, n, BLOCK):
        sc = block_scores(variant, s)
        t = torch.topk(sc, min(k, sc.shape[1]), dim=1)
        cat_s, cat_i = torch.cat([best_s, t.values], 1), torch.cat([best_i, t.indices + s], 1)
        m = torch.topk(cat_s, min(k, cat_s.shape[1]), dim=1)
        best_s, best_i = m.values, torch.gather(cat_i, 1, m.indices)
    return best_i.numpy(), best_s.numpy()


def rescore(rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    out_i, out_s = np.empty((len(rows), k), np.int64), np.empty((len(rows), k), np.float32)
    for qi, r in enumerate(rows):
        exact = vecs[r].astype(np.float32) @ queries[qi].numpy()
        o = np.argsort(-exact, kind="stable")[:k]
        out_i[qi], out_s[qi] = r[o], exact[o]
    return out_i, out_s


bm25 = {r["question_id"]: r["document_ids"] for r in read_jsonl(ROOT / "runs/bm25_own/candidates.jsonl")}
report = {"chunks": n, "dim": dim}
for variant in ("fp16", "int8", "binary", "binary_rs"):
    t0 = time.perf_counter()
    if variant == "binary_rs":
        rows, _ = search("binary", CHUNK_LIMIT * OVERSAMPLE)
        rows, sims = rescore(rows, CHUNK_LIMIT)
    else:
        rows, sims = search(variant, CHUNK_LIMIT)
    secs = time.perf_counter() - t0
    name = f"dense_qwen3_{variant}"
    (ROOT / "runs" / name).mkdir(parents=True, exist_ok=True)
    hyb = ROOT / "runs" / f"hybrid_rrf_k60_{variant}"
    hyb.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "runs" / name / "candidates.jsonl", "w", encoding="utf8") as cand, \
         open(ROOT / "runs" / name / "retrieval.jsonl", "w", encoding="utf8") as ret, \
         open(hyb / "retrieval.jsonl", "w", encoding="utf8") as hret:
        for qid, r, sc in zip(qids, rows, sims):
            best: dict[str, float] = {}
            for d, v in zip(doc_ids[r], sc):
                best.setdefault(d, float(v))
                if len(best) == DEPTH:
                    break
            docs = list(best)
            cand.write(json.dumps({"question_id": qid, "document_ids": docs, "scores": list(best.values())}) + "\n")
            ret.write(json.dumps({"question_id": qid, "document_ids": docs[:TOP_K]}) + "\n")
            fused = rrf([bm25[qid], docs], k=60, top_n=TOP_K)
            hret.write(json.dumps({"question_id": qid, "document_ids": [d for d, _ in fused]}) + "\n")
    report[variant] = {"cpu_seconds_500q": round(secs, 1)}
    print(variant, f"{secs:.1f}s", flush=True)

report["index_bytes"] = {
    "fp16": vecs.nbytes, "int8": codes.nbytes + lo.nbytes + step.nbytes, "binary": signs.nbytes,
    "binary_rs": f"{signs.nbytes} in RAM + {vecs.nbytes} fp16 on disk for rescoring",
}
(ROOT / "runs/quantization.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print(json.dumps(report, indent=2))
