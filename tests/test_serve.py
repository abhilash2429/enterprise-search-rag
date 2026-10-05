import sqlite3
from concurrent.futures import Future
from types import SimpleNamespace

import anyio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from entsearch.docstore import DocStore
from entsearch.retrieval.dense import CHUNK_LIMIT, DenseIndex


def write_index(tmp_path, vecs, doc_ids, shard_rows):
    start = 0
    for i, n in enumerate(shard_rows):
        np.save(tmp_path / f"shard_{i:03d}.npy", vecs[start : start + n].astype(np.float16))
        pq.write_table(pa.table({"doc_id": doc_ids[start : start + n], "chunk_idx": list(range(n))}), tmp_path / f"shard_{i:03d}.parquet")
        start += n


@pytest.fixture
def index(tmp_path):
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((300, 64)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    doc_ids = [f"d{i // 3}" for i in range(300)]  # 3 chunks per doc
    write_index(tmp_path, vecs, doc_ids, [120, 180])
    source = {f"d{j}": ("slack" if j % 2 else "jira") for j in range(100)}
    return tmp_path, vecs.astype(np.float16).astype(np.float32), np.array(doc_ids), source


def maxp(scores, doc_ids, depth):
    best = {}
    for i in np.argsort(-scores, kind="stable"):
        best.setdefault(doc_ids[i], float(scores[i]))
    return list(best)[:depth]


def test_exact_search_is_maxp_over_chunks(index):
    path, vecs, doc_ids, source = index
    q = vecs[7] + 0.1 * vecs[50]
    got = [d for d, _ in DenseIndex(path, source).search(q, depth=20)]
    assert got == maxp(vecs @ q, doc_ids, 20)
    assert got[0] == "d2"  # chunk 7 belongs to doc 2


def test_source_mask_only_returns_routed_sources(index):
    path, vecs, doc_ids, source = index
    got = DenseIndex(path, source).search(vecs[7], depth=30, sources=["slack"])
    assert got and all(source[d] == "slack" for d, _ in got)
    keep = np.array([source[d] == "slack" for d in doc_ids])
    s = vecs @ vecs[7]
    s[~keep] = -np.inf
    assert [d for d, _ in got] == maxp(s, doc_ids, 30)


@pytest.mark.parametrize("binary", [False, True])
def test_search_many_equals_separate_searches(index, binary):
    path, vecs, _, source = index
    idx = DenseIndex(path, source, binary=binary)
    filters = [None, ["slack"], None]
    assert idx.search_many(vecs[3], 25, filters) == [idx.search(vecs[3], 25, f) for f in filters]


def test_binary_rescore_matches_exact_when_shortlist_covers_corpus(index):
    path, vecs, doc_ids, source = index
    assert 300 < CHUNK_LIMIT  # shortlist holds every chunk, so rescoring makes binary exact
    q = vecs[100]
    exact = DenseIndex(path, source).search(q, depth=50)
    binary = DenseIndex(path, source, binary=True).search(q, depth=50)
    assert [d for d, _ in binary] == [d for d, _ in exact]
    np.testing.assert_allclose([s for _, s in binary], [s for _, s in exact], rtol=1e-5)


def test_docstore_get_and_sources(tmp_path):
    p = tmp_path / "docs.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE docs (doc_id TEXT PRIMARY KEY, source_type TEXT, title TEXT, content TEXT)")
    con.executemany("INSERT INTO docs VALUES (?, ?, ?, ?)", [("a", "jira", "A", "alpha"), ("b", "slack", "B", "beta")])
    con.commit()
    con.close()
    store = DocStore(p)
    assert store.get(["b", "missing", "a"]) == {"a": ("jira", "A", "alpha"), "b": ("slack", "B", "beta")}
    assert store.get([]) == {}
    assert store.sources().to_dict() == {"a": "jira", "b": "slack"}


class FakePipeline:
    def search(self, query, k):
        return SimpleNamespace(hits=[{"doc_id": "a"}][:k], query=query)

    def answer(self, question):
        return SimpleNamespace(answer=f"re: {question} [1]", citations=[{"n": 1, "doc_id": "a"}])

    def get_document(self, doc_id):
        return {"doc_id": doc_id, "content": "alpha"} if doc_id == "a" else None


def call(server, name, args):
    return anyio.run(lambda: server.call_tool(name, args))


@pytest.fixture
def server(monkeypatch):
    pytest.importorskip("mcp")
    from entsearch.serve import server as srv

    monkeypatch.setattr(srv, "asdict", lambda o: vars(o))
    fut = Future()
    fut.set_result(FakePipeline())
    return srv.build_server(fut)


def test_server_lists_three_tools(server):
    names = {t.name for t in anyio.run(server.list_tools)}
    assert names == {"search", "answer", "get_document"}


def test_server_tools_return_structured_results(server):
    assert call(server, "search", {"query": "vpn", "k": 1}).structured_content["hits"] == [{"doc_id": "a"}]
    assert call(server, "answer", {"question": "why"}).structured_content["answer"] == "re: why [1]"
    assert call(server, "get_document", {"doc_id": "a"}).structured_content["content"] == "alpha"


def test_server_rejects_bad_k_and_unknown_doc(server):
    # In process, call_tool raises ToolError; the protocol layer turns it into an is_error result for the client.
    from mcp.server.mcpserver.exceptions import ToolError

    with pytest.raises(ToolError, match="k must be"):
        call(server, "search", {"query": "vpn", "k": 0})
    with pytest.raises(ToolError, match="unknown doc_id"):
        call(server, "get_document", {"doc_id": "zzz"})


def test_server_reports_failed_load():
    pytest.importorskip("mcp")
    from entsearch.serve.server import build_server

    from mcp.server.mcpserver.exceptions import ToolError

    fut = Future()
    fut.set_exception(FileNotFoundError("docstore.sqlite missing"))
    with pytest.raises(ToolError, match="index failed to load: docstore.sqlite missing"):
        call(build_server(fut), "search", {"query": "x"})


@pytest.mark.parametrize("verdict, top, flagged, low", [
    ("supported", 0.99, False, False),
    ("supported", 0.50, True, True),
    ("unsupported", 0.99, True, False),
])
def test_confidence_flags_on_verdict_or_low_rerank_score(monkeypatch, verdict, top, flagged, low):
    import json

    import entsearch.serve.pipeline as P
    from entsearch.llm import Generation

    reply = json.dumps({"claims": [], "addresses_question": True, "verdict": verdict, "reasoning": "r"})
    monkeypatch.setattr(P.llm, "bridge_client", lambda: None)
    monkeypatch.setattr(P.llm, "generate", lambda *a, **k: Generation(reply, 10, 5, 0, False, "gpt-6-luna"))
    hits = [P.Hit("a", "slack", "t", "s", top, 0.1)]
    docs = {"a": ("slack", "t", "text")}
    conf, cost = P.Pipeline._confidence(object.__new__(P.Pipeline), "q?", "ans [1]", ["a"], docs, [], hits)
    assert conf["flagged"] is flagged and conf["low_retrieval_score"] is low and conf["verdict"] == verdict
    assert cost > 0


def test_confidence_reports_unparsable_verifier_reply(monkeypatch):
    import entsearch.serve.pipeline as P
    from entsearch.llm import Generation

    monkeypatch.setattr(P.llm, "bridge_client", lambda: None)
    monkeypatch.setattr(P.llm, "generate", lambda *a, **k: Generation("not json", 10, 5, 0, False, "gpt-6-luna"))
    hits = [P.Hit("a", "slack", "t", "s", 0.99, 0.1)]
    conf, _ = P.Pipeline._confidence(object.__new__(P.Pipeline), "q?", "ans", ["a"], {"a": ("slack", "t", "x")}, [], hits)
    assert conf["flagged"] is None and "did not parse" in conf["error"]
