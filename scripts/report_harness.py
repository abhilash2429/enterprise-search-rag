"""Markdown table of harness metrics for one or more runs, per question type, next to the paper's BM25 + GPT-5.4 numbers."""
import argparse
import json

import pandas as pd

from entsearch.answer import read_jsonl
from entsearch.data import ROOT

# Paper, BM25 + GPT-5.4: per-type correctness and document recall.
PAPER_BM25 = {
    "basic": (79.4, 77.7), "semantic": (44.8, 43.2), "intra_document_reasoning": (85.0, 90.0),
    "project_related": (60.0, 65.5), "constrained": (76.7, 85.0), "conflicting_info": (90.0, 82.5),
    "completeness": (40.0, 46.5), "miscellaneous": (85.0, 90.0), "high_level": (50.0, None),
    "info_not_found": (100.0, None), "overall": (68.8, 68.4),
}

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
args = p.parse_args()

rows = []
for run in args.runs:
    d = json.loads((ROOT / "runs" / run / "eval_results.json").read_text(encoding="utf8"))
    stats = {**d["question_type_stats"], "overall": {**d["aggregate_stats"], "count": d["aggregate_stats"]["completed_questions"]}}
    usage = pd.DataFrame(read_jsonl(ROOT / "runs" / run / "usage.jsonl"))
    fresh = usage[~usage.get("cached", pd.Series(False, index=usage.index)).fillna(False).astype(bool)]
    for qtype, s in stats.items():
        rows.append({
            "run": run, "type": qtype, "n": s["count"],
            "correct": s["average_correctness_pct"], "complete": s["average_completeness_pct"],
            "overall": s["combined_correctness_completeness_score"], "recall": s["average_recall_pct"],
            "paper_correct": PAPER_BM25[qtype][0], "paper_recall": PAPER_BM25[qtype][1],
        })
    print(f"{run}: ${fresh.cost_usd.sum():.2f} total, ${fresh.cost_usd.sum() / stats['overall']['count']:.4f}/question")

print()
print(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".1f"))
