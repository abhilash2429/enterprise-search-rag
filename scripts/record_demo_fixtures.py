"""Record real runs of the demo questions into web/fixtures/ for the frontend's offline mode (docs/demo-api.md).

Runs the headline pipeline in process with a fresh, empty LLM cache, so router, answer and verifier calls are real
and the recorded timings are cold. Every event is checked against the contract's keys before anything is written.

  python scripts/record_demo_fixtures.py
  python scripts/record_demo_fixtures.py --only qst_0492     # record some, keep the other recorded runs
"""
import argparse
import json
import tempfile
import time
from pathlib import Path

from entsearch.data import ROOT
from entsearch.serve.demo import demo_questions
from entsearch.serve.pipeline import LISTS, Config, Pipeline

OUT = ROOT / "web/fixtures"
KEYS = {
    "start": {"question", "config", "stages"},
    "route": {"sources", "seconds"},
    "retrieve": {"lists", "seconds"},
    "fuse": {"candidates", "seconds"},
    "rerank": {"hits", "order", "seconds"},
    "generate": {"answer", "abstained", "partial", "citations", "context", "version_pairs", "seconds", "cost_usd"},
    "verify": {"confidence", "skipped", "seconds", "cost_usd"},
    "done": {"seconds", "cost_usd"},
}


def check(events: list[dict]) -> None:
    names = [e["event"] for e in events]
    stages = events[0]["data"]["stages"]
    assert names == ["start", *stages, "done"], names
    for e in events:
        missing = KEYS[e["event"]] - e["data"].keys()
        assert not missing, (e["event"], missing)
    by = {e["event"]: e["data"] for e in events}
    assert set(by["retrieve"]["lists"]) == set(LISTS)
    assert len(by["fuse"]["candidates"]) == 100 and len(by["rerank"]["order"]) == 100 and len(by["rerank"]["hits"]) == 10
    assert [c["n"] for c in by["generate"]["context"]] == list(range(1, 11))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", help="question ids to (re)record; others keep their existing fixtures")
    only = p.parse_args().only
    questions = demo_questions()
    with tempfile.TemporaryDirectory() as cache:
        cfg = Config(ROOT / "data", cache_dir=Path(cache))
        pipe = Pipeline(cfg)
        runs, doc_ids = {}, set()
        for q in questions:
            if only and q["question_id"] not in only:
                continue
            events, t0 = [], time.perf_counter()

            def emit(event: str, payload: dict) -> None:
                events.append({"t": round(time.perf_counter() - t0, 3), "event": event, "data": json.loads(json.dumps(payload))})

            pipe.answer(q["question"], emit)
            check(events)
            runs[q["question_id"]] = events
            by = {e["event"]: e["data"] for e in events}
            doc_ids |= {h["doc_id"] for h in by["rerank"]["hits"]}
            conf = by["verify"]["confidence"] or {}
            print(f"{q['question_id']} {q['question_type']:17s} total {by['done']['seconds']['total']:5.1f}s "
                  f"abstained={by['generate']['abstained']} partial={by['generate']['partial']} "
                  f"cited={[c['n'] for c in by['generate']['citations']]} flagged={conf.get('flagged')} "
                  f"verdict={conf.get('verdict')} ${by['done']['cost_usd']:.4f}", flush=True)

        (OUT / "ask").mkdir(parents=True, exist_ok=True)
        (OUT / "documents").mkdir(parents=True, exist_ok=True)
        for qid, events in runs.items():
            (OUT / "ask" / f"{qid}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf8")
        for d in sorted(doc_ids):
            (OUT / "documents" / f"{d}.json").write_text(json.dumps(pipe.get_document(d), indent=1), encoding="utf8")
        (OUT / "questions.json").write_text(json.dumps(questions, indent=1), encoding="utf8")
        (OUT / "health.json").write_text(json.dumps({"status": "ready", "busy": False, "config": cfg.describe(), "error": None}, indent=1),
                                         encoding="utf8")
    missing = [q["question_id"] for q in questions if not (OUT / "ask" / f"{q['question_id']}.jsonl").exists()]
    assert not missing, f"questions.json lists questions with no recorded run: {missing}"
    print(f"wrote {len(runs)} runs and {len(doc_ids)} documents to {OUT}")


if __name__ == "__main__":
    main()
