"""Dense retrieval mirroring the harness vector baseline: top-100 chunks by cosine, deduped to the top-10 documents."""
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from entsearch.data import ROOT

INDEX = ROOT / "data/index/dense_qwen3-0.6b_c512"
CHUNK_LIMIT, TOP_K = 100, 10

shards = sorted(INDEX.glob("shard_*.npy"))
vecs = torch.from_numpy(np.concatenate([np.load(s) for s in shards]))
doc_ids = np.concatenate([pq.read_table(s.with_suffix(".parquet"), columns=["doc_id"])["doc_id"].to_numpy() for s in shards])
assert len(doc_ids) == len(vecs) == 1_538_921, (len(doc_ids), len(vecs))
device = "cuda" if torch.cuda.is_available() else "cpu"
vecs = vecs.to(device) if device == "cuda" else vecs.float()

queries = torch.from_numpy(np.load(INDEX / "queries.npy")).to(vecs.device, vecs.dtype)
qids = json.loads((INDEX / "queries.json").read_text())

run_dir = ROOT / "runs/dense_qwen3"
run_dir.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
with open(run_dir / "retrieval.jsonl", "w", encoding="utf8") as f:
    for start in range(0, len(qids), 50):
        scores = queries[start : start + 50] @ vecs.T
        top = torch.topk(scores.float(), CHUNK_LIMIT, dim=1).indices.cpu().numpy()
        for qid, row in zip(qids[start : start + 50], top):
            docs = list(dict.fromkeys(doc_ids[row]))[:TOP_K]
            f.write(json.dumps({"question_id": qid, "document_ids": docs}) + "\n")
print(f"{len(qids)} queries in {time.perf_counter() - t0:.1f}s on {device}")
