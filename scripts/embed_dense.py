"""Embed corpus chunks and questions with Qwen3-Embedding-0.6B via vLLM. Runs on the GPU box, resumable per shard.

Output (OUT):
  queries.npy, queries.json        question embeddings and their question_id order
  shard_XXX.npy, shard_XXX.parquet chunk embeddings (fp16, L2-normalized) and (doc_id, chunk_idx)
"""

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
SHARD_DOCS = 20_000
EXPECTED_CHUNKS = 1_538_921
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


def embed(llm, texts: list[str]) -> np.ndarray:
    e = np.asarray([o.outputs.embedding for o in llm.embed(texts, use_tqdm=False)], dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e.astype(np.float16)


def main() -> None:
    from vllm import LLM

    ensure_data()
    OUT.mkdir(parents=True, exist_ok=True)
    llm = LLM(model=MODEL, runner="pooling", max_model_len=2048)

    if not (OUT / "queries.npy").exists():
        q = load_questions()
        np.save(OUT / "queries.npy", embed(llm, [QUERY_TEMPLATE.format(q=x) for x in q.question]))
        (OUT / "queries.json").write_text(json.dumps(list(q.question_id)))

    done_chunks, t_start = 0, time.time()
    for i, batch in enumerate(iter_corpus(batch_size=SHARD_DOCS, columns=("doc_id", "title", "content"))):
        npy, meta = OUT / f"shard_{i:03d}.npy", OUT / f"shard_{i:03d}.parquet"
        if npy.exists() and meta.exists():
            done_chunks += pq.read_metadata(meta).num_rows
            continue
        ids, idx, texts = [], [], []
        for r in batch:
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
            f"total {done_chunks:,}/{EXPECTED_CHUNKS:,} eta {(EXPECTED_CHUNKS - done_chunks) / rate / 3600:.2f}h "
            f"elapsed {(time.time() - t_start) / 3600:.2f}h",
            flush=True,
        )
    print(f"DONE {done_chunks:,} chunks", flush=True)


if __name__ == "__main__":
    main()
