# enterprise-search-rag

Retrieval-augmented question answering over [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
(Onyx, arXiv 2605.05253, MIT): 511,958 synthetic company documents from 9 sources (Slack, email, tickets, docs, ...) and 500 questions.

Work in progress. The goal is hybrid retrieval (BM25 + dense + reranking) with cited answers and abstention, measured with
ablations, paired bootstrap CIs, and an LLM judge validated against hand labels.

## Results so far

| System | Recall@10 | Correctness | Completeness | Overall | $/question |
|---|---|---|---|---|---|
| BM25 (OpenSearch, paper config), this repo | 68.4 | 68.0 | 53.3 | 47.2 | 0.0046 |
| BM25 + GPT-5.4, paper | 68.4 | 68.8 | 56.0 | 50.6 | - |

Generator and judge here are `gpt-oss-120b`; the paper uses GPT-5.4 for both, so only recall is directly comparable.
Recall matches the paper in every question type. Judge vs 100 hand labels: TPR 0.91, TNR 0.94.
Details: [results/](results/), decisions: [docs/decisions.md](docs/decisions.md).

## Layout

- `src/entsearch/`: data loading, split, tokenizer, sparse and dense indexes, near-dup detection, LLM client, tracing
- `scripts/`: runnable steps (indexing, retrieval, answering, judging, reports, labeling tool)
- `tests/`: unit tests (BM25 and metrics tests run against stubs until those are hand-written)
- `third_party/erb`: the benchmark harness, fetched by `scripts/fetch_harness.sh` (not committed)

## Setup

```bash
uv sync
bash scripts/fetch_harness.sh
```

The corpus and questions come from the Hugging Face dataset into `data/erb/`.
