"""Turn cached reranker scores into runs: pure rerank at several depths, plus RRF(rerank, hybrid) as an ablation.

Reads runs/<scores_run>/scores.jsonl (candidates in hybrid order). Pointwise scores, so depth d = re-sort of the
first d candidates. Writes runs/<scores_run>_d<d>/retrieval.jsonl and runs/<scores_run>_rrf_d100/retrieval.jsonl,
only for questions that have scores (partial runs evaluate on what is done).
Ties in reranker score keep hybrid order.
"""
import argparse
import json

import numpy as np

from entsearch.answer import read_jsonl
from entsearch.data import ROOT
from entsearch.retrieval.rrf import rrf

DEPTHS = (20, 50, 100)
TOP_K = 10

p = argparse.ArgumentParser()
p.add_argument("--scores-run", default="rerank_qwen3")
args = p.parse_args()

rows = read_jsonl(ROOT / "runs" / args.scores_run / "scores.jsonl")
runs: dict[str, list[dict]] = {f"{args.scores_run}_d{d}": [] for d in DEPTHS} | {f"{args.scores_run}_rrf_d100": []}
for r in rows:
    ids, s = r["document_ids"], np.asarray(r["scores"])
    for d in DEPTHS:
        order = np.argsort(-s[:d], kind="stable")[:TOP_K]
        runs[f"{args.scores_run}_d{d}"].append({
            "question_id": r["question_id"], "document_ids": [ids[i] for i in order], "rerank_scores": s[order].tolist(),
        })
    reranked = [ids[i] for i in np.argsort(-s, kind="stable")]
    fused = rrf([reranked, ids], top_n=TOP_K)
    runs[f"{args.scores_run}_rrf_d100"].append({"question_id": r["question_id"], "document_ids": [d for d, _ in fused]})

for name, out in runs.items():
    (ROOT / "runs" / name).mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / name / "retrieval.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out), encoding="utf8")
print(f"{len(rows)} questions scored; wrote {', '.join(runs)}")
