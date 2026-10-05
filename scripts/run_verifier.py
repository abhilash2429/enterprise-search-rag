"""Verify a run's answers with the luna verifier (via the local bridge). Seed 0 answers only, cited answerer runs.

Each answer is checked against exactly the context it was generated from (answers.jsonl document_ids and version
pairs). Refusals are not verified. Resumable: writes runs/<run>/verify.jsonl, one row per question:
{"question_id", "verdict", "addresses_question", "flagged", "claims", "reasoning", "cost_usd", ...}, or
{"question_id", "skipped": "abstained"}, or {"question_id", "error", "raw"} when the reply does not parse.
Stops starting new questions once this run's verifier spend reaches --budget.

  python scripts/run_verifier.py rerank_router_d100 --split dev
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from entsearch import llm, verifier
from entsearch.answer import read_jsonl
from entsearch.data import ROOT, load_questions
from entsearch.docstore import DocStore

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--split", choices=["dev", "test"], required=True)
p.add_argument("--workers", type=int, default=8)
p.add_argument("--budget", type=float, default=3.0, help="USD cap on verifier spend for this run")
p.add_argument("--limit", type=int, help="first N questions (smoke test)")
args = p.parse_args()

run_dir = ROOT / "runs" / args.run
out = run_dir / "verify.jsonl"
questions = load_questions(args.split).set_index("question_id").question
answers = [a for a in read_jsonl(run_dir / "answers.jsonl") if a["question_id"] in questions.index]
if not answers or "answer_cited" not in answers[0]:
    raise SystemExit(f"{args.run} has no cited-answerer answers for {args.split}")
rows = read_jsonl(out)
done = {r["question_id"] for r in rows if "error" not in r}
spent = sum(r.get("cost_usd", 0.0) for r in rows)
todo = [a for a in answers if a["question_id"] not in done][: args.limit]
print(f"{len(answers)} answers, {len(done)} done, {len(todo)} to verify, ${spent:.3f} spent so far")

docs = DocStore(ROOT / "data/index/docstore.sqlite")
client = llm.bridge_client()
lock = threading.Lock()


def verify(a: dict) -> None:
    global spent
    with lock:
        if spent >= args.budget:
            return
    qid = a["question_id"]
    if a["abstained"]:
        row = {"question_id": qid, "skipped": "abstained"}
    else:
        ids = a["document_ids"]
        prompt = verifier.build_prompt(
            questions[qid], a["answer_cited"], ids, docs.get(ids), [(x, y, 0.0) for x, y in a["version_pairs"]]
        )
        gen = llm.generate(client, prompt, model=verifier.MODEL, effort="medium")
        usage = {"input_tokens": gen.input_tokens, "output_tokens": gen.output_tokens, "cost_usd": 0.0 if gen.cached else gen.cost}
        try:
            v = verifier.parse(gen.text)
            row = {"question_id": qid, "verdict": v.verdict, "addresses_question": v.addresses_question, "flagged": v.flagged,
                   "claims": v.claims, "reasoning": v.reasoning, **usage}
        except ValueError as e:
            row = {"question_id": qid, "error": str(e), "raw": gen.text, **usage}
    with lock:
        spent += row.get("cost_usd", 0.0)
        with open(out, "a", encoding="utf8") as f:
            f.write(json.dumps(row) + "\n")


with ThreadPoolExecutor(args.workers) as pool:
    list(pool.map(verify, todo))

rows = {r["question_id"]: r for r in read_jsonl(out)}
errors = sum("error" in r for r in rows.values())
flagged = sum(r.get("flagged", False) for r in rows.values())
print(f"{len(rows)} rows, {flagged} flagged, {errors} parse errors, ${spent:.3f} spent")
