import json
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from entsearch.data import HARNESS


@contextmanager
def harness_cwd():
    """The harness imports `src.*` and opens files relative to its root."""
    prev = os.getcwd()
    os.chdir(HARNESS)
    sys.path.insert(0, str(HARNESS))
    try:
        yield
    finally:
        sys.path.remove(str(HARNESS))
        os.chdir(prev)


def harness_import(module: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    return __import__(module, fromlist=["_"])


def write_questions(questions: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf8") as f:
        for row in questions.to_dict("records"):
            f.write(json.dumps(row) + "\n")


def metrics_eval(run_dir: Path, questions: pd.DataFrame, parallelism: int = 8) -> dict:
    """Runs harness metrics_based_eval --no-correction with the Bedrock judge.

    Judge token usage is appended to run_dir/usage.jsonl (stage=judge) by wrapping the SDK call,
    since the harness does not record it.
    """
    from openai.resources.responses import Responses

    from entsearch import llm

    run_dir = run_dir.resolve()
    write_questions(questions, run_dir / "questions.jsonl")
    os.environ.update({
        "OPENAI_BASE_URL": llm.BASE_URL,
        "LLM_API_KEY": llm.bedrock_token(),
        "LLM_PROVIDER": "openai",
        "LLM_MODEL_NAME": llm.MODEL,
        "PYTHONUTF8": "1",
    })

    lock = threading.Lock()
    orig = Responses.create
    p_in, p_out = llm.PRICE[llm.MODEL]

    def create(self, *a, **kw):
        stream = orig(self, *a, **kw)

        def gen():
            for ev in stream:
                if ev.type == "response.completed":
                    # Accounting must never break judging: an exception here surfaces inside the harness's stream,
                    # which retries and then scores the answer as incorrect.
                    try:
                        u = llm.completed_usage(ev)
                        rec = {
                            "stage": "judge", "model": llm.MODEL,
                            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                            "reasoning_tokens": u.reasoning_tokens,
                            "cost_usd": (u.input_tokens * p_in + u.output_tokens * p_out) / 1e6,
                        }
                    except Exception as e:
                        rec = {"stage": "judge", "model": llm.MODEL, "error": repr(e)[:500]}
                    with lock, open(run_dir / "usage.jsonl", "a", encoding="utf8") as f:
                        f.write(json.dumps(rec) + "\n")
                yield ev

        return gen()

    Responses.create = create
    try:
        with harness_cwd():
            from src.scripts.answer_evaluation import metrics_based_eval

            sys.argv = [
                "metrics_based_eval",
                "--answers-file", str(run_dir / "answers.jsonl"),
                "--questions-file", str(run_dir / "questions.jsonl"),
                "--results-file", str(run_dir / "eval_results.json"),
                "--parallelism", str(parallelism),
                "--no-correction",
                "--resume",
            ]
            metrics_based_eval.main()
    finally:
        Responses.create = orig
    return json.loads((run_dir / "eval_results.json").read_text(encoding="utf8"))
