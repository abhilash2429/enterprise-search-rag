"""Deep ranked doc lists per retriever for all 500 questions, input to fusion and reranking.

  bm25   own BM25, top-100 docs
  dense  Qwen3 c512, top-4000 chunks by cosine, each doc ranked by its best chunk (MaxP), top-100 docs
         (--index/--run pick another embedding index, e.g. dense_qwen3-0.6b_whole -> dense_qwen3_whole)
Writes runs/<run>/candidates.jsonl: {"question_id", "document_ids", "scores"}, best first.
"""
import argparse
import json

import numpy as np

from entsearch.data import ROOT, load_questions

DEPTH = 100
CHUNK_LIMIT = 4000
EXPECTED_ROWS = {"dense_qwen3-0.6b_c512": 1_538_921, "dense_qwen3-0.6b_whole": 511_958}


def bm25() -> None:
    from entsearch.index.sparse import SparseIndex
    from entsearch.retrieval.bm25 import BM25

    ix = SparseIndex.load()
    model = BM25().fit(ix.tf, ix.doc_len)
    out = ROOT / "runs/bm25_own/candidates.jsonl"
    with open(out, "w", encoding="utf8") as f:
        for row in load_questions().itertuples():
            idx, scores = model.topk(ix.query_terms(row.question), k=DEPTH)
            f.write(json.dumps({"question_id": row.question_id, "document_ids": [ix.doc_ids[i] for i in idx], "scores": scores.tolist()}) + "\n")
    print(f"wrote {out}")


def dense(index_name: str, run: str) -> None:
    import pyarrow.parquet as pq
    import torch

    index = ROOT / "data/index" / index_name
    shards = sorted(index.glob("shard_*.npy"))
    vecs = torch.from_numpy(np.concatenate([np.load(s) for s in shards])).cuda()
    doc_ids = np.concatenate([pq.read_table(s.with_suffix(".parquet"), columns=["doc_id"])["doc_id"].to_numpy() for s in shards])
    assert len(doc_ids) == len(vecs) == EXPECTED_ROWS[index_name], (len(doc_ids), len(vecs))
    queries = torch.from_numpy(np.load(index / "queries.npy")).to(vecs.device, vecs.dtype)
    qids = json.loads((index / "queries.json").read_text())

    out = ROOT / "runs" / run / "candidates.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    fewest = DEPTH
    with open(out, "w", encoding="utf8") as f:
        for start in range(0, len(qids), 50):
            top = torch.topk((queries[start : start + 50] @ vecs.T).float(), CHUNK_LIMIT, dim=1)
            for qid, rows, sims in zip(qids[start : start + 50], top.indices.cpu().numpy(), top.values.cpu().numpy()):
                best: dict[str, float] = {}
                for d, s in zip(doc_ids[rows], sims):
                    best.setdefault(d, float(s))
                    if len(best) == DEPTH:
                        break
                fewest = min(fewest, len(best))
                f.write(json.dumps({"question_id": qid, "document_ids": list(best), "scores": list(best.values())}) + "\n")
    print(f"wrote {out}; fewest docs for a query: {fewest}")


p = argparse.ArgumentParser()
p.add_argument("retriever", choices=["bm25", "dense"])
p.add_argument("--index", default="dense_qwen3-0.6b_c512")
p.add_argument("--run", default="dense_qwen3")
a = p.parse_args()
bm25() if a.retriever == "bm25" else dense(a.index, a.run)
