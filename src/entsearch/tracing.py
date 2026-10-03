"""OpenTelemetry tracing: every stage is a span with latency, tokens, dollars and gold-hit attribution.

Spans always go to runs/<run>/spans.jsonl (the source for cost/latency analysis). They also go to Langfuse
over OTLP when LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST are set (.env is read).
"""

import base64
import json
import os
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from entsearch.data import ROOT

_provider: TracerProvider | None = None


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        rows = []
        for s in spans:
            rows.append(json.dumps({
                "name": s.name,
                "trace_id": f"{s.context.trace_id:032x}",
                "span_id": f"{s.context.span_id:016x}",
                "parent_id": f"{s.parent.span_id:016x}" if s.parent else None,
                "start_ns": s.start_time,
                "duration_ms": (s.end_time - s.start_time) / 1e6,
                "status": s.status.status_code.name,
                "attributes": dict(s.attributes or {}),
            }))
        with self._lock, open(self.path, "a", encoding="utf8") as f:
            f.write("".join(r + "\n" for r in rows))
        return SpanExportResult.SUCCESS


def _langfuse_exporter():
    load_dotenv(ROOT / ".env")
    pk, sk, host = (os.getenv(k) for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"))
    if not (pk and sk and host):
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    return OTLPSpanExporter(
        endpoint=f"{host.rstrip('/')}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
    )


def init(run: str) -> bool:
    """Configure tracing for a run. Returns True if Langfuse export is on. Call once per process."""
    global _provider
    if _provider is not None:
        return False
    _provider = TracerProvider(resource=Resource.create({"service.name": "entsearch", "entsearch.run": run}))
    _provider.add_span_processor(BatchSpanProcessor(JsonlSpanExporter(ROOT / "runs" / run / "spans.jsonl")))
    lf = _langfuse_exporter()
    if lf is not None:
        _provider.add_span_processor(BatchSpanProcessor(lf))
    trace.set_tracer_provider(_provider)
    return lf is not None


def shutdown() -> None:
    if _provider is not None:
        _provider.shutdown()


def tracer():
    return trace.get_tracer("entsearch")


@contextmanager
def span(name: str, **attrs):
    """Stage span. With no init(), this is a no-op span from the default provider."""
    with tracer().start_as_current_span(name) as s:
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(k, v)
        yield s


def question_span(run: str, question_id: str, question_type: str | None = None):
    return span(
        "question",
        **{
            "langfuse.trace.name": "question",
            "langfuse.session.id": run,
            "entsearch.question_id": question_id,
            "entsearch.question_type": question_type,
            "langfuse.trace.metadata.question_id": question_id,
        },
    )


def record_llm(s, model: str, input_tokens: int, output_tokens: int, reasoning_tokens: int, cost_usd: float, cached: bool) -> None:
    s.set_attribute("langfuse.observation.type", "generation")
    s.set_attribute("gen_ai.request.model", model)
    s.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    s.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    s.set_attribute("gen_ai.usage.reasoning_tokens", reasoning_tokens)
    s.set_attribute("gen_ai.usage.cost", cost_usd)
    s.set_attribute("langfuse.observation.usage_details", json.dumps({"input": input_tokens, "output": output_tokens}))
    s.set_attribute("langfuse.observation.cost_details", json.dumps({"total": cost_usd}))
    s.set_attribute("entsearch.cached", cached)


def record_hits(s, retrieved: Iterable[str], gold: Iterable[str]) -> None:
    """Hit attribution: how many gold docs survive this stage's output."""
    retrieved, gold = list(retrieved), set(gold)
    s.set_attribute("retrieval.k", len(retrieved))
    s.set_attribute("retrieval.gold_total", len(gold))
    s.set_attribute("retrieval.gold_hits", len(gold & set(retrieved)))
