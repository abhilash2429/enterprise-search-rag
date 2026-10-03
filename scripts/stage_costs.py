"""Per-stage latency (p50/p95), dollars and gold survival from a run's spans.jsonl."""
import argparse
import json

import pandas as pd

from entsearch.answer import read_jsonl
from entsearch.data import ROOT

p = argparse.ArgumentParser()
p.add_argument("run")
args = p.parse_args()

rows = []
for r in read_jsonl(ROOT / "runs" / args.run / "spans.jsonl"):
    a = r["attributes"]
    rows.append({
        "stage": r["name"], "ms": r["duration_ms"], "cost": a.get("gen_ai.usage.cost", 0.0),
        "cached": a.get("entsearch.cached", False),
        "gold_hits": a.get("retrieval.gold_hits"), "gold_total": a.get("retrieval.gold_total"),
    })
df = pd.DataFrame(rows)
g = df.groupby("stage")
out = pd.DataFrame({
    "n": g.size(),
    "p50_ms": g.ms.median(),
    "p95_ms": g.ms.quantile(0.95),
    "usd_per_q": g.cost.mean(),
    "gold_survival": g.gold_hits.sum() / g.gold_total.sum().where(lambda s: s > 0),
})
print(out.round(4).to_string())
if df.cached.any():
    print(f"\nnote: {int(df.cached.sum())} spans were LLM cache hits; their latency is not real generation time")
