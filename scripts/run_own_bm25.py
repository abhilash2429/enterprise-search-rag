"""Hand-written BM25 over the sparse index, top-10 for all 500 questions."""
import json
import time

from tqdm import tqdm

from entsearch.data import ROOT, load_questions
from entsearch.index.sparse import SparseIndex
from entsearch.retrieval.bm25 import BM25

ix = SparseIndex.load()
t0 = time.perf_counter()
model = BM25().fit(ix.tf, ix.doc_len)
print(f"fit {time.perf_counter() - t0:.1f}s")

run_dir = ROOT / "runs/bm25_own"
run_dir.mkdir(parents=True, exist_ok=True)
lat = []
with open(run_dir / "retrieval.jsonl", "w", encoding="utf8") as f:
    for row in tqdm(load_questions().itertuples(), total=500, desc="search"):
        t0 = time.perf_counter()
        idx, _ = model.topk(ix.query_terms(row.question), k=10)
        lat.append(time.perf_counter() - t0)
        f.write(json.dumps({"question_id": row.question_id, "document_ids": [ix.doc_ids[i] for i in idx]}) + "\n")
lat.sort()
print(f"latency p50 {lat[len(lat) // 2] * 1e3:.1f} ms  p95 {lat[int(len(lat) * 0.95)] * 1e3:.1f} ms")
