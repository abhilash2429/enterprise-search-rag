"""Harness-equivalent answer generation: retrieved doc ids -> ANSWER_GEN_PROMPT -> answers.jsonl."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from entsearch import answerer as cited
from entsearch import llm, tracing
from entsearch.data import load_docs
from entsearch.harness import harness_import


def format_context(doc_ids: list[str], docs: dict[str, tuple[str, str]]) -> str:
    """Mirror of harness src/utils/retrieval.format_context_documents."""
    return "\n\n".join(
        f"--- Document {i} (ID: {d}) ---\nTitle: {docs[d][0]}\n\n{docs[d][1]}" for i, d in enumerate(doc_ids, 1)
    )


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]


def answer_run(run_dir: Path, questions: pd.DataFrame, workers: int = 8, answerer: str = "harness", seed: int = 0) -> None:
    """Reads run_dir/retrieval.jsonl, writes run_dir/answers.jsonl (harness format) and run_dir/usage.jsonl. Resumable.

    answerer="harness": the harness's ANSWER_GEN_PROMPT, the baseline. answerer="cited": entsearch.answerer, where the
    judged `answer` has citation markers stripped and the row also carries the cited text, cited doc ids, abstention
    flags, and the top reranker score when retrieval.jsonl has one.
    """
    if answerer not in ("harness", "cited"):
        raise ValueError(f"unknown answerer {answerer!r}")
    prompt_tpl = harness_import("src.prompts.vector_search_answer_gen").ANSWER_GEN_PROMPT
    retrieval = {r["question_id"]: r for r in read_jsonl(run_dir / "retrieval.jsonl")}
    retrieved = {q: r["document_ids"] for q, r in retrieval.items()}
    done = {r["question_id"] for r in read_jsonl(run_dir / "answers.jsonl")}
    todo = questions[~questions.question_id.isin(done)]
    missing = set(todo.question_id) - set(retrieved)
    if missing:
        raise ValueError(f"{len(missing)} questions have no retrieval rows, e.g. {sorted(missing)[:3]}")

    docs = load_docs([d for q in todo.question_id for d in retrieved[q]], with_source=answerer == "cited")
    versions = None
    if answerer == "cited":
        from entsearch.index.neardup_embed import VersionPairs  # pulls in torch; only the cited answerer needs it

        versions = VersionPairs()
    client = llm.client()
    lock = threading.Lock()

    def one(row) -> None:
        ids = retrieved[row.question_id]
        extra = {}
        if answerer == "cited":
            pairs = versions(ids)
            prompt = cited.build_prompt(row.question, ids, docs, pairs)
            extra["version_pairs"] = [[a, b] for a, b, _ in pairs]
            if "rerank_scores" in retrieval[row.question_id]:
                extra["rerank_top"] = retrieval[row.question_id]["rerank_scores"][0] if ids else None
        else:
            prompt = prompt_tpl.format(context_documents=format_context(ids, docs), question=row.question)
        with tracing.question_span(run_dir.name, row.question_id, row.question_type) as root:
            root.set_attribute("langfuse.observation.input", json.dumps(row.question))
            with tracing.span("generate") as s:
                tracing.record_hits(s, ids, row.expected_doc_ids)
                gen = llm.generate(client, prompt, seed=seed)
                tracing.record_llm(s, llm.MODEL, gen.input_tokens, gen.output_tokens, gen.reasoning_tokens, gen.cost, gen.cached)
                s.set_attribute("langfuse.observation.output", json.dumps(gen.text))
        answer = gen.text
        if answerer == "cited":
            p = cited.parse(gen.text, ids)
            answer = p.text
            extra |= {
                "answer_cited": gen.text, "cited_doc_ids": p.cited_doc_ids, "invalid_citations": p.invalid_citations,
                "abstained": p.abstained, "partial": p.partial,
            }
        with lock:
            with open(run_dir / "answers.jsonl", "a", encoding="utf8") as f:
                f.write(json.dumps({"question_id": row.question_id, "answer": answer, "document_ids": ids} | extra) + "\n")
            with open(run_dir / "usage.jsonl", "a", encoding="utf8") as f:
                f.write(json.dumps({
                    "question_id": row.question_id, "stage": "generate", "model": llm.MODEL,
                    "input_tokens": gen.input_tokens, "output_tokens": gen.output_tokens,
                    "reasoning_tokens": gen.reasoning_tokens, "cost_usd": gen.cost, "cached": gen.cached,
                }) + "\n")

    failed = []
    with ThreadPoolExecutor(workers) as ex:
        futures = {ex.submit(one, row): row.question_id for row in todo.itertuples()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="answer"):
            try:
                fut.result()
            except Exception as e:
                failed.append((futures[fut], repr(e)))
    if failed:
        raise RuntimeError(f"{len(failed)} questions failed; rerun to resume. First: {failed[0]}")
