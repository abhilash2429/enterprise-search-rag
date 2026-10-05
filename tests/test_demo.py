import json
import threading
from concurrent.futures import Future

import pytest
from starlette.testclient import TestClient

from entsearch.serve.demo import DEMO_QUESTIONS, build_app


class FakePipeline:
    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate

    def answer(self, question, emit):
        emit("start", {"question": question, "stages": ["retrieve", "generate"]})
        if self.gate:
            self.gate.wait(5)
        if question == "boom":
            raise RuntimeError("model fell over")
        emit("generate", {"answer": "42 [1]"})
        emit("done", {"seconds": {"total": 0.1}, "cost_usd": 0.0})

    def get_document(self, doc_id):
        return {"doc_id": doc_id, "source": "slack", "title": "t", "content": "c"} if doc_id == "d1" else None


def client(pipe=None, loading=False):
    fut: Future = Future()
    if not loading:
        fut.set_result(pipe or FakePipeline())
    return TestClient(build_app(fut, {"router": True}))


def events(text: str) -> list[tuple[str, dict]]:
    out, event = [], None
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            out.append((event, json.loads(line.split(":", 1)[1])))
    return out


def test_health_and_questions():
    c = client()
    assert c.get("/api/health").json() == {"status": "ready", "busy": False, "config": {"router": True}, "error": None}
    qs = c.get("/api/questions").json()
    assert [q["question_id"] for q in qs] == DEMO_QUESTIONS
    assert all(q["question"] and q["gold_answer"] and q["question_type"] for q in qs)


def test_ask_streams_pipeline_events_in_order():
    with client().stream("GET", "/api/ask", params={"q": "what?"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        got = events(r.read().decode())
    assert [e for e, _ in got] == ["start", "generate", "done"]
    assert got[0][1]["question"] == "what?" and got[1][1]["answer"] == "42 [1]"


def test_ask_reports_pipeline_failure_as_error_event():
    with client().stream("GET", "/api/ask", params={"q": "boom"}) as r:
        got = events(r.read().decode())
    assert got[-1] == ("error", {"message": "RuntimeError: model fell over"})


def test_second_ask_while_busy_is_409_and_health_says_busy():
    gate = threading.Event()
    c = client(FakePipeline(gate))
    first = threading.Thread(target=lambda: c.get("/api/ask", params={"q": "slow"}))
    first.start()
    for _ in range(100):
        if c.get("/api/health").json()["busy"]:
            break
        threading.Event().wait(0.02)
    assert c.get("/api/health").json()["busy"]
    assert c.get("/api/ask", params={"q": "again"}).status_code == 409
    gate.set()
    first.join(5)
    assert not c.get("/api/health").json()["busy"]


@pytest.mark.parametrize("path, status", [("/api/ask?q=x", 503), ("/api/documents/d1", 503)])
def test_loading_returns_503(path, status):
    c = client(loading=True)
    assert c.get("/api/health").json()["status"] == "loading"
    assert c.get(path).status_code == status


def test_documents_and_bad_requests():
    c = client()
    assert c.get("/api/documents/d1").json()["title"] == "t"
    assert c.get("/api/documents/nope").status_code == 404
    assert c.get("/api/ask").status_code == 400


def test_cors_allows_the_next_dev_server():
    r = client().get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
