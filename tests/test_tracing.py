import json
import shutil

from entsearch import tracing
from entsearch.data import ROOT


def test_spans_written_with_hierarchy_and_attributes(monkeypatch):
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.setenv(k, "")
    run = "_test_tracing"
    path = ROOT / "runs" / run
    shutil.rmtree(path, ignore_errors=True)
    try:
        assert tracing.init(run) is False
        with tracing.question_span(run, "qst_x", "basic"):
            with tracing.span("generate") as s:
                tracing.record_hits(s, ["a", "b", "c"], ["b", "z"])
                tracing.record_llm(s, "m", 100, 20, 5, 0.01, False)
        tracing.shutdown()
        spans = {r["name"]: r for r in map(json.loads, (path / "spans.jsonl").read_text().splitlines())}
        q, g = spans["question"], spans["generate"]
        assert g["parent_id"] == q["span_id"] and g["trace_id"] == q["trace_id"] and q["parent_id"] is None
        assert g["attributes"]["retrieval.gold_hits"] == 1 and g["attributes"]["retrieval.gold_total"] == 2
        assert g["attributes"]["gen_ai.usage.cost"] == 0.01
        assert q["attributes"]["entsearch.question_id"] == "qst_x"
    finally:
        shutil.rmtree(path, ignore_errors=True)
