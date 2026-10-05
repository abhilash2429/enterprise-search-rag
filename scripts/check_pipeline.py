"""Online pipeline vs the evaluated batch run: same top-10 per dev question? Also per-stage latency.

  python scripts/check_pipeline.py --n 50                      # headline config vs runs/rerank_router_d100
  python scripts/check_pipeline.py --n 50 --no-rerank --dense binary --device cpu --against hybrid_router

Exact agreement is not expected everywhere: the batch dense search scored on GPU in fp16 (ties at the top-4000 chunk
cutoff break arbitrarily) and reranker scores carry fp16 batch-shape noise. BM25, the router (cached LLM response),
and RRF are deterministic.
"""
import argparse
import json

import numpy as np

from entsearch.answer import read_jsonl
from entsearch.data import ROOT, load_questions
from entsearch.metrics import recall_at_k
from entsearch.serve.pipeline import Config, Pipeline

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=50)
p.add_argument("--against", default="rerank_router_d100", help="run whose retrieval.jsonl is the reference")
p.add_argument("--no-router", action="store_true")
p.add_argument("--no-rerank", action="store_true")
p.add_argument("--dense", choices=["fp16", "binary", "none"], default="fp16")
p.add_argument("--device", default="cuda")
a = p.parse_args()

ref = {r["question_id"]: r["document_ids"][:10] for r in read_jsonl(ROOT / "runs" / a.against / "retrieval.jsonl")}
q = load_questions("dev").head(a.n)
pipe = Pipeline(Config(ROOT / "data", router=not a.no_router, rerank=not a.no_rerank, dense=a.dense, device=a.device))

rows = []
for x in q.itertuples():
    res = pipe.search(x.question, k=10)
    got = [h.doc_id for h in res.hits]
    want = ref[x.question_id]
    gold = x.expected_doc_ids
    rows.append({
        "question_id": x.question_id, "same": got == want, "overlap": len(set(got) & set(want)) / 10,
        "recall_online": recall_at_k(got, gold, 10) if gold else None,
        "recall_batch": recall_at_k(want, gold, 10) if gold else None, **{f"s_{k}": v for k, v in res.seconds.items()},
    })

out = ROOT / "runs" / f"check_pipeline_{a.against}.jsonl"
out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
same = np.mean([r["same"] for r in rows])
overlap = np.mean([r["overlap"] for r in rows])
ro = np.mean([r["recall_online"] for r in rows if r["recall_online"] is not None])
rb = np.mean([r["recall_batch"] for r in rows if r["recall_batch"] is not None])
print(f"n={len(rows)} identical top-10 {same:.0%}, mean overlap@10 {overlap:.3f}, recall@10 online {100*ro:.1f} vs batch {100*rb:.1f}")
for k in [k for k in rows[0] if k.startswith("s_")]:
    v = np.array([r[k] for r in rows if k in r])
    print(f"  {k[2:]:9s} p50 {np.median(v):.2f}s  p90 {np.percentile(v, 90):.2f}s")
print(f"wrote {out}")
