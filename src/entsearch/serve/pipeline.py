"""Single-query serving path for the headline config: source router -> hybrid RRF -> rerank -> cited answer.

Mirrors the batch scripts step for step (run_router.py retrieve, run_rerank.py, rerank_runs.py, answer.py cited), so
one query through here should land on the same documents as the evaluated runs. Each result records which config
served it and per-stage latency.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from entsearch import answerer, llm, verifier
from entsearch.data import doc_text
from entsearch.docstore import DocStore
from entsearch.retrieval import router
from entsearch.retrieval.rrf import rrf

DEPTH, RRF_K, SNIPPET = 100, 60, 300


@dataclass
class Config:
    data_dir: Path
    router: bool = True
    rerank: bool = True
    dense: str = "fp16"  # fp16 | binary | none
    device: str = "cuda"
    # Padded tokens per rerank batch. Half the batch scripts' 16384: on Windows (WDDM) every CUDA allocation is backed
    # by host commit, so activation memory costs the same in RAM; scores differ only by fp16 batch-shape noise.
    rerank_token_budget: int = 8192
    cache_dir: Path | None = None  # LLM response cache; default <data_dir>/../cache/llm
    verify: bool = True  # confidence flag: gpt-6-luna verifier through the local bridge + reranker top-score threshold

    def describe(self) -> dict:
        return {"router": self.router, "rerank": self.rerank, "dense": self.dense, "verify": self.verify}


@dataclass
class Hit:
    doc_id: str
    source: str
    title: str
    snippet: str
    rerank_score: float | None
    fused_score: float


@dataclass
class SearchResult:
    hits: list[Hit]
    routed_sources: list[str]
    config: dict
    seconds: dict[str, float] = field(default_factory=dict)


@dataclass
class AnswerResult:
    answer: str  # with [n] citation markers
    citations: list[dict]  # n, doc_id, source, title for every context doc the answer cites
    abstained: bool
    partial: bool
    context_doc_ids: list[str]
    routed_sources: list[str]
    config: dict
    cost_usd: float
    seconds: dict[str, float] = field(default_factory=dict)
    # Flag only: the answer is returned either way. None when verify is off or the answer is a refusal.
    confidence: dict | None = None


class Pipeline:
    def __init__(self, cfg: Config):
        from entsearch.index.neardup_embed import VersionPairs
        from entsearch.index.sparse import SparseIndex
        from entsearch.retrieval.bm25 import BM25

        self.cfg = cfg
        index = cfg.data_dir / "index"
        llm.CACHE = cfg.cache_dir or cfg.data_dir.parent / "cache/llm"
        self.docs = DocStore(index / "docstore.sqlite")
        doc_source = self.docs.sources()

        self.sparse = SparseIndex.load(index / "sparse")
        self.bm25 = BM25().fit(self.sparse.tf, self.sparse.doc_len)
        self.sparse.tf = None  # BM25 keeps its own indices and weights; the raw counts are 1.4 GB serving never reads
        self.source_names = list(doc_source.cat.categories)
        self.bm25_src = doc_source.reindex(self.sparse.doc_ids).cat.codes.to_numpy().astype(np.int8)
        if (self.bm25_src < 0).any():
            raise ValueError("sparse index holds docs missing from the doc store; rebuild one of them")

        self.dense = self.encoder = self.reranker = None
        if cfg.dense != "none":
            from entsearch.retrieval.dense import DenseIndex, QueryEncoder

            self.dense = DenseIndex(index / "dense_qwen3-0.6b_c512", doc_source, binary=cfg.dense == "binary")
            self.encoder = QueryEncoder(device=cfg.device)
        if cfg.rerank:
            from entsearch.retrieval.rerank import Reranker

            self.reranker = Reranker(token_budget=cfg.rerank_token_budget)
        self.versions = VersionPairs(out=index / "neardup_embed")

    def _release_gpu_cache(self) -> None:
        """Return the reranker's cached activation memory to the driver between queries (host commit on WDDM)."""
        import torch

        torch.cuda.empty_cache()

    def _bm25(self, query: str, sources: list[str] | None = None) -> list[str]:
        mask = None
        if sources:
            mask = np.isin(self.bm25_src, [self.source_names.index(s) for s in sources if s in self.source_names])
        idx, _ = self.bm25.topk(self.sparse.query_terms(query), k=DEPTH, mask=mask)
        return [self.sparse.doc_ids[i] for i in idx]

    def route(self, query: str) -> tuple[list[str], float]:
        gen = llm.generate(llm.client(), router.build_prompt(query), effort="low")
        return router.parse(gen.text), 0.0 if gen.cached else gen.cost

    def search(self, query: str, k: int = 10) -> SearchResult:
        return self._search(query, k)[0]

    def _search(self, query: str, k: int) -> tuple[SearchResult, float]:
        t, cost = {}, 0.0
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            t[name] = round(now - clock, 3)
            clock = now

        routed: list[str] = []
        if self.cfg.router:
            routed, cost = self.route(query)
            lap("route")
        filters = [None, routed] if routed else [None]
        dense = [[] for _ in filters]
        if self.dense is not None:
            dense = self.dense.search_many(self.encoder(query), DEPTH, filters)
        lists = [l for f, hits in zip(filters, dense) for l in (self._bm25(query, f), [d for d, _ in hits])]
        lap("retrieve")
        fused = rrf([l for l in lists if l], k=RRF_K, top_n=DEPTH)
        ids = [d for d, _ in fused]
        fused_score = dict(fused)
        docs = self.docs.get(ids)
        lap("fetch")

        rerank: dict[str, float] = {}
        if self.reranker is not None and ids:
            scores = self.reranker.score(query, [doc_text(docs[d][1], docs[d][2]) for d in ids])
            ids = [ids[i] for i in np.argsort(-scores, kind="stable")]
            rerank = dict(zip([d for d, _ in fused], scores.tolist()))
            self._release_gpu_cache()
            lap("rerank")

        hits = [
            Hit(d, docs[d][0], docs[d][1], docs[d][2][:SNIPPET], rerank.get(d), fused_score[d])
            for d in ids[:k]
        ]
        return SearchResult(hits, routed, self.cfg.describe(), t), cost

    def answer(self, question: str) -> AnswerResult:
        start = time.perf_counter()
        res, cost = self._search(question, k=10)
        ids = [h.doc_id for h in res.hits]
        docs = self.docs.get(ids)
        t = dict(res.seconds)
        clock = time.perf_counter()
        pairs = self.versions(ids)
        gen = llm.generate(llm.client(), answerer.build_prompt(question, ids, docs, pairs))
        t["generate"] = round(time.perf_counter() - clock, 3)
        p = answerer.parse(gen.text, ids)
        confidence = None
        if self.cfg.verify and not p.abstained:
            clock = time.perf_counter()
            confidence, vcost = self._confidence(question, gen.text, ids, docs, pairs, res.hits)
            cost += vcost
            t["verify"] = round(time.perf_counter() - clock, 3)
        t["total"] = round(time.perf_counter() - start, 3)
        num = {d: n for n, d in enumerate(ids, 1)}
        citations = [{"n": num[d], "doc_id": d, "source": docs[d][0], "title": docs[d][1]} for d in p.cited_doc_ids]
        return AnswerResult(
            answer=gen.text, citations=citations, abstained=p.abstained, partial=p.partial, context_doc_ids=ids,
            routed_sources=res.routed_sources, config=res.config,
            cost_usd=round(cost + (0.0 if gen.cached else gen.cost), 6), seconds=t, confidence=confidence,
        )

    def _confidence(self, question, answer, ids, docs, pairs, hits) -> tuple[dict, float]:
        gen = llm.generate(llm.bridge_client(), verifier.build_prompt(question, answer, ids, docs, pairs),
                           model=verifier.MODEL, effort="medium")
        cost = 0.0 if gen.cached else gen.cost
        top = hits[0].rerank_score if hits else None
        low = top is not None and top < verifier.FLAG_TAU
        try:
            v = verifier.parse(gen.text)
        except ValueError as e:
            return {"flagged": None, "error": f"verifier reply did not parse: {e}", "rerank_top": top}, cost
        return {
            "flagged": v.flagged or low, "verdict": v.verdict, "addresses_question": v.addresses_question,
            "low_retrieval_score": low, "rerank_top": top, "threshold": verifier.FLAG_TAU,
            "claims": v.claims, "reasoning": v.reasoning,
        }, cost

    def get_document(self, doc_id: str) -> dict | None:
        d = self.docs.get([doc_id]).get(doc_id)
        return None if d is None else {"doc_id": doc_id, "source": d[0], "title": d[1], "content": d[2]}
