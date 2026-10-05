"""MCP server over the EnterpriseRAG-Bench indexes: search, answer, get_document.

  entsearch-mcp --data-dir /path/to/repo/data                 # stdio, headline config (router + hybrid + GPU rerank)
  entsearch-mcp --no-rerank --dense binary --device cpu       # CPU only, 197 MB dense index in RAM
  entsearch-mcp --transport streamable-http --port 8000

Indexes load in a background thread so the client handshake is not blocked; the first tool call waits for them.
Logs go to stderr: on stdio, stdout carries the protocol.
"""

import argparse
import logging
import sys
import threading
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from entsearch.data import ROOT
from entsearch.serve.pipeline import Config, Pipeline

log = logging.getLogger("entsearch.mcp")

INSTRUCTIONS = """Search and question answering over a company's internal documents (512K documents from Slack, \
Gmail, Google Drive, Confluence, Jira, Linear, GitHub, HubSpot and Fireflies meeting transcripts; the \
EnterpriseRAG-Bench corpus). Use `answer` for a direct cited answer, `search` to see ranked documents yourself, and \
`get_document` to read a full document from either."""


def build_server(load: "Future[Pipeline]") -> MCPServer:
    """Tools over a pipeline that may still be loading. Calls are serialized: the GPU models are not thread-safe."""
    server = MCPServer("entsearch", instructions=INSTRUCTIONS)
    lock = threading.Lock()

    def pipeline() -> Pipeline:
        try:
            return load.result()
        except Exception as e:
            raise ToolError(f"index failed to load: {e}") from e

    @server.tool(structured_output=True)
    def search(query: str, k: int = 10) -> dict[str, Any]:
        """Rank documents for a query: an LLM router picks likely sources, BM25 and dense retrieval (plus copies
        restricted to the routed sources) are fused with RRF, and a cross-encoder reranks the top 100.

        Returns the top k (1 to 50) documents with doc_id, source, title, a snippet and scores, plus the routed
        sources, the serving config and per-stage seconds. Read a full document with get_document."""
        if not 1 <= k <= 50:
            raise ToolError("k must be between 1 and 50")
        with lock:
            return asdict(pipeline().search(query, k))

    @server.tool(structured_output=True)
    def answer(question: str) -> dict[str, Any]:
        """Answer a question from the top 10 retrieved documents. The answer cites documents with [n] markers;
        `citations` maps each n to its doc_id, source and title. If the documents do not contain the answer it says
        so (abstained=true) rather than guessing; partial=true means some of the question is not covered."""
        with lock:
            return asdict(pipeline().answer(question))

    @server.tool(structured_output=True)
    def get_document(doc_id: str) -> dict[str, Any]:
        """Full text of one document by doc_id (as returned by search or answer)."""
        doc = pipeline().get_document(doc_id)
        if doc is None:
            raise ToolError(f"unknown doc_id {doc_id!r}")
        return doc

    return server


def start_loading(cfg: Config) -> "Future[Pipeline]":
    fut: Future[Pipeline] = Future()

    def run() -> None:
        try:
            log.info("loading indexes from %s (%s)", cfg.data_dir, cfg.describe())
            fut.set_result(Pipeline(cfg))
            log.info("ready")
        except Exception as e:
            log.exception("index load failed")
            fut.set_exception(e)

    threading.Thread(target=run, name="entsearch-load", daemon=True).start()
    return fut


def main() -> None:
    p = argparse.ArgumentParser(prog="entsearch-mcp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data", help="directory holding index/ (default: repo data/)")
    p.add_argument("--no-router", action="store_true", help="skip the LLM source router")
    p.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder; return fused order")
    p.add_argument("--dense", choices=["fp16", "binary", "none"], default="fp16")
    p.add_argument("--device", default="cuda", help="device for the query encoder and reranker")
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")

    cfg = Config(data_dir=a.data_dir, router=not a.no_router, rerank=not a.no_rerank, dense=a.dense, device=a.device)
    server = build_server(start_loading(cfg))
    if a.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=a.host, port=a.port)


if __name__ == "__main__":
    main()
