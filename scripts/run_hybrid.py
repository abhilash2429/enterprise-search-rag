"""RRF over bm25_own and dense_qwen3 candidates (100 docs each), for each k in the sweep.

Writes runs/hybrid_rrf_k<k>/retrieval.jsonl (top-10 docs, harness format) and candidates.jsonl (fused top-100,
the reranker's input). k=60 is the headline config; the other k values are an ablation.
"""
import json

from entsearch.answer import read_jsonl
from entsearch.data import ROOT
from entsearch.retrieval.rrf import rrf

KS = (10, 20, 60, 100)
TOP_K, DEPTH = 10, 100

lists = [{r["question_id"]: r["document_ids"] for r in read_jsonl(ROOT / "runs" / run / "candidates.jsonl")} for run in ("bm25_own", "dense_qwen3")]
assert lists[0].keys() == lists[1].keys()

for k in KS:
    run_dir = ROOT / f"runs/hybrid_rrf_k{k}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "retrieval.jsonl", "w", encoding="utf8") as ret, open(run_dir / "candidates.jsonl", "w", encoding="utf8") as cand:
        for qid in lists[0]:
            fused = rrf([l[qid] for l in lists], k=k, top_n=DEPTH)
            docs = [d for d, _ in fused]
            ret.write(json.dumps({"question_id": qid, "document_ids": docs[:TOP_K]}) + "\n")
            cand.write(json.dumps({"question_id": qid, "document_ids": docs, "scores": [s for _, s in fused]}) + "\n")
    print(f"wrote {run_dir}")
