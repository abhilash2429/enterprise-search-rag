"""Generate answers for a retrieval run with gpt-oss-120b, then judge with the harness (--no-correction)."""
import argparse
import shutil

import pandas as pd

from entsearch.answer import answer_run, read_jsonl
from entsearch.data import ROOT, load_questions
from entsearch import tracing
from entsearch.harness import metrics_eval

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--split", choices=["dev", "test"], default=None, help="default: all 500")
p.add_argument("--limit", type=int, default=None, help="first N questions (stratified order not guaranteed)")
p.add_argument("--workers", type=int, default=8)
p.add_argument("--skip-eval", action="store_true")
p.add_argument("--answerer", choices=["harness", "cited"], default="harness")
p.add_argument("--seed", type=int, default=0, help="generation replicate; seed k > 0 writes runs/<run>_s<k>, reusing <run>'s retrieval")
args = p.parse_args()

run_dir = ROOT / "runs" / args.run
if args.seed:
    base, run_dir = run_dir, ROOT / "runs" / f"{args.run}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "retrieval.jsonl").exists():
        shutil.copy(base / "retrieval.jsonl", run_dir / "retrieval.jsonl")
q = load_questions(args.split)
if args.limit:
    q = q.groupby("question_type", group_keys=False).head(max(1, args.limit // q.question_type.nunique())).head(args.limit)

print("langfuse export on" if tracing.init(run_dir.name) else "langfuse export off (no keys); spans -> runs/<run>/spans.jsonl")
try:
    answer_run(run_dir, q, workers=args.workers, answerer=args.answerer, seed=args.seed)
finally:
    tracing.shutdown()
if not args.skip_eval:
    metrics_eval(run_dir, q, parallelism=args.workers)

usage = pd.DataFrame(read_jsonl(run_dir / "usage.jsonl"))
fresh = usage[~usage.get("cached", pd.Series(False, index=usage.index)).fillna(False).astype(bool)]
print(fresh.groupby("stage")[["input_tokens", "output_tokens", "cost_usd"]].sum().to_string())
print(f"questions: {len(q)}  total cost ${fresh.cost_usd.sum():.3f}  per question ${fresh.cost_usd.sum() / len(q):.4f}")
