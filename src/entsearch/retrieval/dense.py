"""Online dense retrieval over a Qwen3-Embedding index: query encoding and per-query MaxP doc search.

The batch scripts (dump_candidates.py, run_router.py) score all 500 questions at once from precomputed query vectors;
this is the single-query path for serving. Same model, same template, same MaxP over the top-4000 chunks.
"""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Model-card default task description and format; not tuned on this benchmark.
QUERY_TEMPLATE = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:{q}"
CHUNK_LIMIT = 4000
OVERSAMPLE = 4


class QueryEncoder:
    """L2-normalized last-token embedding of the templated query. No EOS is appended, matching how vLLM tokenized
    the text prompts that built the index."""

    def __init__(self, device: str = "cuda", model: str = MODEL):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model, dtype=dtype, device_map=device).eval()

    def __call__(self, query: str) -> np.ndarray:
        ids = self.tok(QUERY_TEMPLATE.format(q=query), return_tensors="pt").input_ids.to(self.device)
        with self.torch.inference_mode():
            h = self.model(input_ids=ids).last_hidden_state[0, -1].float()
        return (h / h.norm()).cpu().numpy()


class DenseIndex:
    """Chunk vectors (fp16 shards) with doc ids; search returns docs ranked by their best chunk.

    The fp16 vectors stay memory-mapped (file-backed pages the OS can drop and reread, not process memory): exact
    search streams them per query. binary=True also keeps sign bits in RAM (1 bit per dimension), ranks chunks by
    Hamming distance, and rescores an OVERSAMPLE x shortlist with the mapped fp16 vectors. Chunk-to-doc is an int32
    code per chunk, not a string.
    """

    def __init__(self, path: Path, doc_source: Mapping[str, str], binary: bool = False):
        """doc_source: doc_id -> source_type for every doc in the index (a dict, or a Series indexed by doc_id)."""
        shards = sorted(path.glob("shard_*.npy"))
        if not shards:
            raise FileNotFoundError(f"no shard_*.npy in {path}")
        self.binary = binary
        self.shards = [np.load(s, mmap_mode="r") for s in shards]
        self.offsets = np.cumsum([0] + [len(m) for m in self.shards])
        ids = pa.chunked_array([pq.read_table(s.with_suffix(".parquet"), columns=["doc_id"])["doc_id"] for s in shards])
        enc = ids.combine_chunks().dictionary_encode()
        self.chunk_doc = enc.indices.to_numpy()
        self.doc_ids = enc.dictionary.to_pylist()
        src = pd.Series(doc_source).astype("category")
        doc_src = src.reindex(self.doc_ids)
        if doc_src.isna().any():
            raise ValueError(f"{int(doc_src.isna().sum())} indexed docs have no source, e.g. {doc_src[doc_src.isna()].index[0]}")
        self.sources = list(src.cat.categories)
        self.chunk_src = doc_src.cat.codes.to_numpy().astype(np.int8)[self.chunk_doc]
        if binary:
            self.signs = np.concatenate([np.packbits(np.asarray(m) > 0, axis=1) for m in self.shards])

    def _exact(self, q: np.ndarray) -> np.ndarray:
        out = np.empty(self.offsets[-1], dtype=np.float32)
        for m, start in zip(self.shards, self.offsets):
            for s in range(0, len(m), 100_000):
                block = np.asarray(m[s : s + 100_000]).astype(np.float32)
                out[start + s : start + s + len(block)] = block @ q
        return out

    def _rows(self, idx: np.ndarray) -> np.ndarray:
        shard = np.searchsorted(self.offsets, idx, side="right") - 1
        return np.stack([self.shards[s][i - self.offsets[s]] for s, i in zip(shard, idx)]).astype(np.float32)

    def search(self, q: np.ndarray, depth: int = 100, sources: list[str] | None = None) -> list[tuple[str, float]]:
        """Top `depth` docs by MaxP score, best first; with `sources`, only chunks from those sources."""
        return self.search_many(q, depth, [sources])[0]

    def search_many(
        self, q: np.ndarray, depth: int, source_sets: list[list[str] | None]
    ) -> list[list[tuple[str, float]]]:
        """search() once per source filter, scoring the index for the query only once."""
        if self.binary:
            qbits = np.packbits(q > 0)
            ham = np.empty(len(self.signs), dtype=np.int32)
            for s in range(0, len(ham), 200_000):
                ham[s : s + 200_000] = np.bitwise_count(self.signs[s : s + 200_000] ^ qbits).sum(1)
            score = -ham.astype(np.float32)
        else:
            score = self._exact(q)
        return [self._rank(q, score, depth, sources) for sources in source_sets]

    def _rank(self, q: np.ndarray, score: np.ndarray, depth: int, sources: list[str] | None) -> list[tuple[str, float]]:
        if sources:
            allowed = np.isin(self.chunk_src, [self.sources.index(s) for s in sources if s in self.sources])
            score = np.where(allowed, score, -np.inf)
        k = min(CHUNK_LIMIT * (OVERSAMPLE if self.binary else 1), len(score))
        top = np.argpartition(-score, k - 1)[:k]
        top = top[np.isfinite(score[top])]
        if self.binary and len(top):
            exact = self._rows(top) @ q
            order = np.argsort(-exact, kind="stable")[:CHUNK_LIMIT]
            top, sims = top[order], exact[order]
        else:
            top = top[np.argsort(-score[top], kind="stable")]
            sims = score[top]
        best: dict[int, float] = {}
        for d, v in zip(self.chunk_doc[top].tolist(), sims):
            best.setdefault(d, float(v))
            if len(best) == depth:
                break
        return [(self.doc_ids[d], v) for d, v in best.items()]
