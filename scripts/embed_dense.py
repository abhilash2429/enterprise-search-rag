"""Embed corpus chunks and questions with Qwen3-Embedding-0.6B via vLLM. Runs on the GPU box, resumable per shard.

--whole-doc embeds each document as one unit (title + content, first WHOLE_DOC_TOKENS tokens) into OUT_WHOLE instead
of 512-token chunks; the chunking ablation.

Output (OUT or OUT_WHOLE):
  queries.npy, queries.json        question embeddings and their question_id order
  shard_XXX.npy, shard_XXX.parquet chunk embeddings (fp16, L2-normalized) and (doc_id, chunk_idx)
"""

import argparse
import hashlib
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from entsearch.data import CORPUS, ROOT, doc_text, iter_corpus, load_questions
from entsearch.index.chunking import CHUNK_SIZE, chunk_text

MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Model-card default task description and format; not tuned on this benchmark.
QUERY_TEMPLATE = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{q}"
OUT = ROOT / "data/index/dense_qwen3-0.6b_c512"
OUT_WHOLE = ROOT / "data/index/dense_qwen3-0.6b_whole"
WHOLE_DOC_TOKENS = 8192
SHARD_DOCS = 20_000
EXPECTED_CHUNKS = 1_538_921
EXPECTED_DOCS = 511_958
CORPUS_SHA256 = "6b0747bf160af9427b12101537d53056ac592ada9831c1a98ae01fa50a8d2a9f"


def ensure_data() -> None:
    from huggingface_hub import hf_hub_download

    for f in ("data/documents/test.parquet", "data/questions/test.parquet"):
        hf_hub_download("onyx-dot-app/EnterpriseRAG-Bench", f, repo_type="dataset", local_dir=ROOT / "data/erb")
    h = hashlib.sha256()
    with open(CORPUS, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    assert h.hexdigest() == CORPUS_SHA256, "corpus parquet differs from the one the project was built on"


def embed(llm, texts: list) -> np.ndarray:
    e = np.asarray([o.outputs.embedding for o in llm.embed(texts, use_tqdm=False)], dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e.astype(np.float16)


def head_tokens(tok, text: str, n: int) -> dict:
    """First n tokens, as a token prompt. vLLM's truncate_prompt_tokens keeps the tail instead, dropping the title.
    No EOS is appended, matching how vLLM tokenizes the c512 text prompts (verified: identical embeddings)."""
    return {"prompt_token_ids": tok(text).input_ids[:n]}


def main() -> None:
    from vllm import LLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--whole-doc", action="store_true")
    whole = ap.parse_args().whole_doc
    out = OUT_WHOLE if whole else OUT

    ensure_data()
    out.mkdir(parents=True, exist_ok=True)
    # Whole docs: vLLM 0.30 pooling with chunked prefill (its default, max_num_batched_tokens = max_model_len) stops
    # scheduling once many 8K prompts are in flight; scheduling each prompt whole avoids the hang.
    kw = {"enable_chunked_prefill": False, "max_num_batched_tokens": 2 * WHOLE_DOC_TOKENS} if whole else {}
    llm = LLM(model=MODEL, runner="pooling", max_model_len=WHOLE_DOC_TOKENS if whole else 2048, **kw)
    tok = llm.get_tokenizer()

    if not (out / "queries.npy").exists():
        q = load_questions()
        np.save(out / "queries.npy", embed(llm, [QUERY_TEMPLATE.format(q=x) for x in q.question]))
        (out / "queries.json").write_text(json.dumps(list(q.question_id)))

    expected = EXPECTED_DOCS if whole else EXPECTED_CHUNKS
    done_chunks, t_start = 0, time.time()
    for i, batch in enumerate(iter_corpus(batch_size=SHARD_DOCS, columns=("doc_id", "title", "content"))):
        npy, meta = out / f"shard_{i:03d}.npy", out / f"shard_{i:03d}.parquet"
        if npy.exists() and meta.exists():
            done_chunks += pq.read_metadata(meta).num_rows
            continue
        ids, idx, texts = [], [], []
        for r in batch:
            if whole:
                ids.append(r["doc_id"])
                idx.append(0)
                texts.append(head_tokens(tok, doc_text(r["title"], r["content"]), WHOLE_DOC_TOKENS))
                continue
            for j, c in enumerate(chunk_text(doc_text(r["title"], r["content"]), CHUNK_SIZE)):
                ids.append(r["doc_id"])
                idx.append(j)
                texts.append(c)
        t0 = time.time()
        vecs = embed(llm, texts)
        np.save(npy.with_suffix(".tmp.npy"), vecs)
        pq.write_table(pa.table({"doc_id": ids, "chunk_idx": idx}), meta)
        npy.with_suffix(".tmp.npy").rename(npy)
        done_chunks += len(texts)
        dt = time.time() - t0
        rate = len(texts) / dt
        print(
            f"shard {i:03d} chunks {len(texts):,} {dt:.0f}s {rate:.0f} chunks/s "
            f"total {done_chunks:,}/{expected:,} eta {(expected - done_chunks) / rate / 3600:.2f}h "
            f"elapsed {(time.time() - t_start) / 3600:.2f}h",
            flush=True,
        )
    print(f"DONE {done_chunks:,} chunks", flush=True)


if __name__ == "__main__":
    main()
