"""Recall@10, MRR@10, nDCG@10 per run: overall, per split, per question_type. Paper BM25 recall alongside."""
import argparse
import json

import pandas as pd

from entsearch.answer import read_jsonl
from entsearch.data import ROOT, SPLITS, load_questions
from entsearch.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

# Paper Table (BM25 + GPT-5.4), document recall % per question type.
PAPER_BM25_RECALL = {
    "basic": 77.7, "semantic": 43.2, "intra_document_reasoning": 90.0, "project_related": 65.5,
    "constrained": 85.0, "conflicting_info": 82.5, "completeness": 46.5, "miscellaneous": 90.0, "overall": 68.4,
}

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--k", type=int, default=10)
p.add_argument("--split", choices=["dev", "test"], help="score only this split (use dev while tuning, to keep test unseen)")
args = p.parse_args()

q = load_questions()
dev = set((SPLITS / "dev.txt").read_text().split())
q["split"] = q.question_id.map(lambda x: "dev" if x in dev else "test")
q = q[q.expected_doc_ids.map(len) > 0]
if args.split:
    q = q[q.split == args.split]

frames = []
for run in args.runs:
    ret = {r["question_id"]: list(dict.fromkeys(r["document_ids"])) for r in read_jsonl(ROOT / "runs" / run / "retrieval.jsonl")}
    rows = []
    for r in q.itertuples():
        got = ret[r.question_id]
        rows.append({
            "run": run, "question_id": r.question_id, "question_type": r.question_type, "split": r.split,
            "recall": recall_at_k(got, r.expected_doc_ids, args.k),
            "mrr": reciprocal_rank(got, r.expected_doc_ids, args.k),
            "ndcg": ndcg_at_k(got, r.expected_doc_ids, args.k),
        })
    per_q = pd.DataFrame(rows)
    per_q.to_csv(ROOT / "runs" / run / f"retrieval_metrics{'_' + args.split if args.split else ''}.csv", index=False)
    frames.append(per_q)

per_q = pd.concat(frames)
by_type = per_q.pivot_table(index="question_type", columns="run", values="recall", aggfunc="mean") * 100
by_type.loc["overall"] = per_q.groupby("run").recall.mean() * 100
if not args.split:  # paper numbers are over all 500, not comparable to one split
    by_type["paper_bm25"] = pd.Series(PAPER_BM25_RECALL)
summary = per_q.groupby(["run", "split"])[["recall", "mrr", "ndcg"]].mean().unstack("split") * 100
summary[("recall", "all")] = per_q.groupby("run").recall.mean() * 100

pd.set_option("display.float_format", "{:.1f}".format)
print(f"recall@{args.k} by question type (n={len(q)} questions with gold)\n{by_type.round(1).to_string()}\n")
print(summary.round(1).to_string())
