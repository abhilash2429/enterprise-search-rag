"""Score the hybrid_rrf_k60 top-100 candidates with Qwen3-Reranker, dev questions first. Resumable.

Writes runs/<run>/scores.jsonl: {"question_id", "document_ids", "scores"} in candidate (hybrid) order.
"""
import argparse
import json
import time

from entsearch.answer import read_jsonl
from entsearch.data import ROOT, doc_text, load_docs, load_questions
from entsearch.retrieval.rerank import DEFAULT_INSTRUCTION, INSTRUCTION, Reranker

p = argparse.ArgumentParser()
p.add_argument("--run", default="rerank_qwen3")
p.add_argument("--instruction", choices=["custom", "default"], default="custom")
p.add_argument("--split", choices=["dev", "test"], help="only this split (default: dev, then test)")
p.add_argument("--limit", type=int, help="stop after this many questions (timing)")
p.add_argument("--token-budget", type=int, default=16_384, help="padded tokens per batch; raise on bigger GPUs")
args = p.parse_args()

cands = {r["question_id"]: r["document_ids"] for r in read_jsonl(ROOT / "runs/hybrid_rrf_k60/candidates.jsonl")}
questions = {s: load_questions(s) for s in ("dev", "test")}
todo = [r for s in ([args.split] if args.split else ["dev", "test"]) for r in questions[s].itertuples()]

run_dir = ROOT / "runs" / args.run
run_dir.mkdir(parents=True, exist_ok=True)
out = run_dir / "scores.jsonl"
done = {r["question_id"] for r in read_jsonl(out)} if out.exists() else set()
todo = [r for r in todo if r.question_id not in done][: args.limit]

docs = load_docs([d for r in todo for d in cands[r.question_id]])
rr = Reranker(INSTRUCTION if args.instruction == "custom" else DEFAULT_INSTRUCTION, token_budget=args.token_budget)
t0 = time.perf_counter()
with open(out, "a", encoding="utf8") as f:
    for i, r in enumerate(todo, 1):
        ids = cands[r.question_id]
        scores = rr.score(r.question, [doc_text(*docs[d]) for d in ids])
        f.write(json.dumps({"question_id": r.question_id, "document_ids": ids, "scores": scores.tolist()}) + "\n")
        f.flush()
        el = time.perf_counter() - t0
        print(f"{i}/{len(todo)}  {el / i:.1f} s/question  eta {el / i * (len(todo) - i) / 60:.0f} min", flush=True)
