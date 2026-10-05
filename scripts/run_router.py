"""Source router ablation.

  predict   gpt-oss routes all 500 questions -> runs/router/routes.jsonl (+ usage.jsonl); prints accuracy vs gold
            source_types on dev
  retrieve  RRF k=60 over [bm25, dense, bm25 restricted to routed sources, dense restricted to routed sources], top-100
            -> runs/hybrid_router/{candidates,retrieval}.jsonl. Unrouted docs keep their bm25/dense ranks, so routing
            is a soft boost. Dense runs on CPU in row blocks.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from entsearch import llm
from entsearch.answer import read_jsonl
from entsearch.data import CORPUS, ROOT, load_questions
from entsearch.retrieval.router import build_prompt, parse
from entsearch.retrieval.rrf import rrf

OUT = ROOT / "runs/router"
DEPTH, TOP_K, CHUNK_LIMIT, BLOCK = 100, 10, 4000, 200_000


def predict(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q = load_questions()
    done = {r["question_id"] for r in read_jsonl(OUT / "routes.jsonl")}
    client, lock = llm.client(), threading.Lock()

    def one(row) -> None:
        gen = llm.generate(client, build_prompt(row.question), effort="low")
        with lock:
            with open(OUT / "routes.jsonl", "a", encoding="utf8") as f:
                f.write(json.dumps({"question_id": row.question_id, "sources": parse(gen.text), "raw": gen.text}) + "\n")
            with open(OUT / "usage.jsonl", "a", encoding="utf8") as f:
                f.write(json.dumps({
                    "question_id": row.question_id, "stage": "route", "model": llm.MODEL,
                    "input_tokens": gen.input_tokens, "output_tokens": gen.output_tokens, "cost_usd": gen.cost,
                    "cached": gen.cached,
                }) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(one, [r for r in q.itertuples() if r.question_id not in done]), total=len(q) - len(done)))

    routes = {r["question_id"]: r["sources"] for r in read_jsonl(OUT / "routes.jsonl")}
    dev = load_questions("dev")
    dev = dev[dev.source_types.map(len) > 0]
    pred = dev.question_id.map(routes)
    gold = dev.source_types.map(set)
    usage = [r for r in read_jsonl(OUT / "usage.jsonl") if not r["cached"]]
    print(f"dev n={len(dev)} (questions with gold sources)")
    print(f"  top-1 in gold      {np.mean([p[:1] and p[0] in g for p, g in zip(pred, gold)]):.3f}")
    print(f"  any gold predicted {np.mean([bool(set(p) & g) for p, g in zip(pred, gold)]):.3f}")
    print(f"  all gold predicted {np.mean([g <= set(p) for p, g in zip(pred, gold)]):.3f}")
    print(f"  mean predicted     {np.mean([len(p) for p in pred]):.2f}; empty {sum(not p for p in routes.values())}")
    print(f"  cost ${sum(r['cost_usd'] for r in usage):.4f} over {len(usage)} uncached calls")


def retrieve() -> None:
    import torch

    from entsearch.index.sparse import SparseIndex
    from entsearch.retrieval.bm25 import BM25

    routes = {r["question_id"]: r["sources"] for r in read_jsonl(OUT / "routes.jsonl")}
    src = pq.read_table(CORPUS, columns=["doc_id", "source_type"]).to_pandas().drop_duplicates("doc_id")
    doc_source = dict(zip(src.doc_id, src.source_type))
    names = sorted(set(doc_source.values()))
    code = {s: i for i, s in enumerate(names)}

    q = load_questions()
    ix = SparseIndex.load()
    bm25 = BM25().fit(ix.tf, ix.doc_len)
    bm25_src = np.array([code[doc_source[d]] for d in ix.doc_ids])
    bm25_routed = {}
    for row in tqdm(q.itertuples(), total=len(q), desc="bm25 routed"):
        allowed = [code[s] for s in routes[row.question_id]]
        if allowed:
            idx, _ = bm25.topk(ix.query_terms(row.question), k=DEPTH, mask=np.isin(bm25_src, allowed))
            bm25_routed[row.question_id] = [ix.doc_ids[i] for i in idx]

    index = ROOT / "data/index/dense_qwen3-0.6b_c512"
    shards = sorted(index.glob("shard_*.npy"))
    vecs = np.concatenate([np.load(s) for s in shards])
    chunk_doc = np.concatenate([pq.read_table(s.with_suffix(".parquet"), columns=["doc_id"])["doc_id"].to_numpy() for s in shards])
    chunk_src = torch.from_numpy(np.array([code[doc_source[d]] for d in chunk_doc]))
    qids = json.loads((index / "queries.json").read_text())
    queries = torch.from_numpy(np.load(index / "queries.npy")).float()
    allow = torch.zeros(len(qids), len(names), dtype=torch.bool)
    for i, qid in enumerate(qids):
        for s in routes[qid]:
            allow[i, code[s]] = True
    best_s = torch.full((len(qids), 0), -np.inf)
    best_i = torch.zeros((len(qids), 0), dtype=torch.long)
    for s in tqdm(range(0, len(vecs), BLOCK), desc="dense routed"):
        sc = queries @ torch.from_numpy(vecs[s : s + BLOCK].astype(np.float32)).T
        sc[~allow[:, chunk_src[s : s + BLOCK]]] = -np.inf
        t = torch.topk(sc, min(CHUNK_LIMIT, sc.shape[1]), dim=1)
        cat_s, cat_i = torch.cat([best_s, t.values], 1), torch.cat([best_i, t.indices + s], 1)
        m = torch.topk(cat_s, CHUNK_LIMIT, dim=1)
        best_s, best_i = m.values, torch.gather(cat_i, 1, m.indices)
    dense_routed = {}
    for qid, rows, sims in zip(qids, best_i.numpy(), best_s.numpy()):
        docs: list[str] = []
        for d, v in zip(chunk_doc[rows], sims):
            if not np.isfinite(v):
                break
            if d not in docs:
                docs.append(d)
                if len(docs) == DEPTH:
                    break
        if docs:
            dense_routed[qid] = docs

    base = [{r["question_id"]: r["document_ids"] for r in read_jsonl(ROOT / "runs" / run / "candidates.jsonl")}
            for run in ("bm25_own", "dense_qwen3")]
    run_dir = ROOT / "runs/hybrid_router"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "retrieval.jsonl", "w", encoding="utf8") as ret, open(run_dir / "candidates.jsonl", "w", encoding="utf8") as cand:
        for qid in base[0]:
            lists = [b[qid] for b in base] + [r[qid] for r in (bm25_routed, dense_routed) if qid in r]
            fused = rrf(lists, k=60, top_n=DEPTH)
            docs = [d for d, _ in fused]
            ret.write(json.dumps({"question_id": qid, "document_ids": docs[:TOP_K]}) + "\n")
            cand.write(json.dumps({"question_id": qid, "document_ids": docs, "scores": [s for _, s in fused]}) + "\n")
    print(f"wrote {run_dir}")


p = argparse.ArgumentParser()
p.add_argument("step", choices=["predict", "retrieve"])
p.add_argument("--workers", type=int, default=8)
a = p.parse_args()
predict(a.workers) if a.step == "predict" else retrieve()
