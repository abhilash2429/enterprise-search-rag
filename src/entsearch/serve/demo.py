"""Demo API: the headline pipeline over HTTP, streaming each stage as a server-sent event, for the recorded demo UI.

  uv run entsearch-demo                    # http://127.0.0.1:8000/api, indexes load in the background

Contract: docs/demo-api.md. One question runs at a time (one GPU): /api/health reports busy, and /api/ask answers 409
while a question is running. A client that disconnects does not stop the run; the GPU work finishes first.
"""

import argparse
import asyncio
import json
import logging
import sys
import threading
from concurrent.futures import Future
from pathlib import Path

from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from entsearch.data import ROOT, load_questions
from entsearch.serve.pipeline import Config, Pipeline

log = logging.getLogger("entsearch.demo")

# Dev questions picked for what each one shows (seed-0 outcomes): lookup, version conflict, semantic, multi-source
# project question, info not found (refusal on seed 0; answered and flagged when recorded), partial answer, flagged wrong
# answer, flagged correct answer.
DEMO_QUESTIONS = ["qst_0147", "qst_0421", "qst_0180", "qst_0343", "qst_0484", "qst_0459", "qst_0037", "qst_0396"]
ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def demo_questions() -> list[dict]:
    q = load_questions().set_index("question_id")
    return [
        {"question_id": i, "question": q.question[i], "question_type": q.question_type[i],
         "gold_answer": q.gold_answer[i], "expected_doc_ids": list(q.expected_doc_ids[i])}
        for i in DEMO_QUESTIONS
    ]


def build_app(load: "Future[Pipeline]", config: dict) -> Starlette:
    busy = threading.Lock()
    questions = demo_questions()

    def state() -> tuple[str, str | None]:
        if not load.done():
            return "loading", None
        e = load.exception()
        return ("error", f"{type(e).__name__}: {e}") if e else ("ready", None)

    async def health(request: Request) -> JSONResponse:
        status, error = state()
        return JSONResponse({"status": status, "busy": busy.locked(), "config": config, "error": error})

    async def list_questions(request: Request) -> JSONResponse:
        return JSONResponse(questions)

    async def document(request: Request) -> JSONResponse:
        status, error = state()
        if status != "ready":
            return JSONResponse({"error": error or "indexes are still loading"}, status_code=503)
        doc = load.result().get_document(request.path_params["doc_id"])
        if doc is None:
            return JSONResponse({"error": f"unknown doc_id {request.path_params['doc_id']}"}, status_code=404)
        return JSONResponse(doc)

    async def ask(request: Request):
        question = request.query_params.get("q", "").strip()
        if not question:
            return JSONResponse({"error": "missing q"}, status_code=400)
        status, error = state()
        if status != "ready":
            return JSONResponse({"error": error or "indexes are still loading"}, status_code=503)
        if not busy.acquire(blocking=False):
            return JSONResponse({"error": "another question is running"}, status_code=409)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def emit(event: str, payload: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event, payload))

        def work() -> None:
            try:
                load.result().answer(question, emit)
            except Exception as e:
                log.exception("ask failed")
                emit("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                busy.release()
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=work, name="entsearch-ask", daemon=True).start()

        async def events():
            while (item := await queue.get()) is not None:
                yield {"event": item[0], "data": json.dumps(item[1])}

        return EventSourceResponse(events())

    return Starlette(
        routes=[
            Route("/api/health", health), Route("/api/questions", list_questions), Route("/api/ask", ask),
            Route("/api/documents/{doc_id}", document),
        ],
        middleware=[Middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET"])],
    )


def main() -> None:
    import uvicorn

    from entsearch.serve.server import start_loading

    p = argparse.ArgumentParser(prog="entsearch-demo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data", help="directory holding index/ (default: repo data/)")
    p.add_argument("--no-verify", action="store_true", help="skip the confidence flag (needs the Azure bridge on :4100)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")
    cfg = Config(data_dir=a.data_dir, verify=not a.no_verify)
    uvicorn.run(build_app(start_loading(cfg), cfg.describe()), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
