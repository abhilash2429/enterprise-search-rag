"""Frontier file agent baseline: the harness's agent_retrieval.py, unmodified, plus per-question usage and a spend cap.

Runs under Linux (WSL here): the harness executes agent commands with shell=True, which is cmd.exe on Windows.
Needs a harness checkout with generated_data/sources materialized, and only the harness's own deps.

  python scripts/run_agent_baseline.py --harness ~/erb-agent/erb --run-dir runs/agent_gpt6_luna \
      --base-url http://<host>:4100/v1 --model gpt-6-luna --parallelism 12 --budget 10

Writes run_dir/answers.jsonl (harness format) and run_dir/usage.jsonl: one row per LLM call (stage=generate,
question_id, tokens, cost_usd) and one per question (stage=agent, seconds, calls). Once spend reaches --budget, no new
question starts; skipped ids go to run_dir/skipped.txt and their placeholder answers are dropped.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

# USD per 1M tokens, Azure Global Standard meters: (uncached input, cached input, output), short and long context.
# Uncached input is priced at the cache-write rate (0.125 vs 0.10 plain), since usage does not say which prefix was
# written. Calls over LONG_CONTEXT input tokens bill at the long-context meters.
PRICE = {"gpt-6-luna": ((0.125, 0.01, 0.50), (0.25, 0.02, 0.75))}
LONG_CONTEXT = 272_000

ap = argparse.ArgumentParser()
ap.add_argument("--harness", type=Path, required=True)
ap.add_argument("--run-dir", type=Path, required=True)
ap.add_argument("--base-url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--parallelism", type=int, default=8)
ap.add_argument("--reasoning-level", default="medium")
ap.add_argument("--budget", type=float, required=True, help="USD; no new question starts once spent")
ap.add_argument("--limit", type=int)
args = ap.parse_args()

run_dir = args.run_dir.resolve()
os.environ.update({
    "OPENAI_BASE_URL": args.base_url, "LLM_API_KEY": "bridge", "LLM_PROVIDER": "openai",
    "LLM_MODEL_NAME": args.model, "CHEAP_LLM_MODEL_NAME": args.model, "PYTHONUTF8": "1",
})
os.chdir(args.harness)
sys.path.insert(0, str(args.harness))

from openai.resources.responses import Responses  # noqa: E402

from src.scripts.answer_generation import agent_retrieval as ar  # noqa: E402

lock = threading.Lock()
local = threading.local()
# usage_aborted.jsonl: calls from questions cut off by a crash, kept out of $/q but still counted against the budget.
spent = sum(
    json.loads(line).get("cost_usd", 0.0)
    for path in (run_dir / "usage.jsonl", run_dir / "usage_aborted.jsonl") if path.exists()
    for line in path.read_text(encoding="utf8").splitlines() if line.strip()
)
skipped: list[str] = []


def log(rec: dict) -> None:
    with lock, open(run_dir / "usage.jsonl", "a", encoding="utf8") as f:
        f.write(json.dumps(rec) + "\n")


def _get(obj, key, default=0):
    if obj is None:
        return default
    v = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return default if v is None else v


orig_create = Responses.create


def create(self, *a, **kw):
    stream = orig_create(self, *a, **kw)
    if not kw.get("stream"):
        return stream

    def gen():
        global spent
        for ev in stream:
            if ev.type == "response.completed":
                try:
                    u = ev.response.usage
                    i, o = _get(u, "input_tokens"), _get(u, "output_tokens")
                    c = _get(_get(u, "input_tokens_details", None), "cached_tokens")
                    p_in, p_cached, p_out = PRICE[args.model][i > LONG_CONTEXT]
                    cost = ((i - c) * p_in + c * p_cached + o * p_out) / 1e6
                    with lock:
                        spent += cost
                        local.calls = getattr(local, "calls", 0) + 1
                    log({
                        "stage": "generate", "question_id": getattr(local, "qid", None), "model": args.model,
                        "input_tokens": i, "cached_tokens": c,
                        "output_tokens": o,
                        "reasoning_tokens": _get(_get(u, "output_tokens_details", None), "reasoning_tokens"),
                        "cost_usd": cost,
                    })
                except Exception as e:  # accounting must not break the agent
                    log({"stage": "generate", "question_id": getattr(local, "qid", None), "error": repr(e)[:500]})
            yield ev

    return gen()


Responses.create = create


class BudgetExhausted(RuntimeError):
    pass


orig_run = ar.run_agent_for_question


def run_agent_for_question(*, question_id, **kw):
    with lock:
        over = spent >= args.budget
        if over:
            skipped.append(question_id)
    if over:
        raise BudgetExhausted(f"${spent:.2f} spent")
    local.qid, local.calls = question_id, 0
    t0 = time.monotonic()
    try:
        return orig_run(question_id=question_id, **kw)
    finally:
        log({"stage": "agent", "question_id": question_id, "seconds": time.monotonic() - t0, "calls": local.calls})


ar.run_agent_for_question = run_agent_for_question
# Non-interactive: keep going if an optional command is missing; use the checkout's committed uuid index.
ar.confirm_yes_no = lambda prompt, default=True: False if "Regenerate" in prompt else default

answers = run_dir / "answers.jsonl"
sys.argv = [
    "agent_retrieval", "--questions-file", str(run_dir / "questions.jsonl"), "--output", str(answers),
    "--parallelism", str(args.parallelism), "--reasoning-level", args.reasoning_level, "--model", args.model, "--resume",
    *(["--limit", str(args.limit)] if args.limit else []),
]
ar.main()

if skipped:
    drop = set(skipped)
    rows = [json.loads(line) for line in answers.read_text(encoding="utf8").splitlines() if line.strip()]
    answers.write_text("".join(json.dumps(r) + "\n" for r in rows if r["question_id"] not in drop), encoding="utf8")
    (run_dir / "skipped.txt").write_text("\n".join(sorted(drop)) + "\n", encoding="utf8")
print(f"spent ${spent:.2f}; skipped {len(skipped)} over budget")
