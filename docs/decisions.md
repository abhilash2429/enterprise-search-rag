# Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-03 | Generator and judge: `openai.gpt-oss-120b` on Bedrock, `metrics_based_eval --no-correction`, reasoning effort medium | gpt-5.4 unaffordable or gated; one judge for every system keeps comparisons fair. Judge scores are compared only between my own systems. |
| 2026-10-03 | Paper BM25 reproduced with OpenSearch (harness settings verbatim) | Isolates generator and judge as the only differences. Recall@10 matched the paper at 68.4 in every question type. |
| 2026-10-03 | Hand-written BM25 runs over a sparse TF matrix with a tokenizer matching Lucene `standard` | Token-for-token identical to OpenSearch on 2,000 sampled docs. |
| 2026-10-03 | BM25 baseline run on all 500 questions | Untuned baseline, so no test leakage; matches the paper's reporting. |
| 2026-10-03 | Dense: Qwen3-Embedding-0.6B, full 1024 dims, paper chunking (512 cl100k tokens, 10% overlap, top-100 chunks to top-10 docs), model-card query instruction | Clean comparison against the paper's vector baseline; other chunkings are ablations. |
| 2026-10-03 | Embeddings computed on one EC2 g6.xlarge (vLLM), searched exactly on the laptop GPU | AWS credits; vLLM vectors verified equal to sentence-transformers (cosine ~1.000). |
| 2026-10-03 | Reranker: Qwen3-Reranker-0.6B | Same family as the embedder; fits the laptop GPU for dev runs. |
| 2026-10-03 | Tracing: Langfuse Cloud (free tier) via OpenTelemetry | No ops; demo can link to traces. |
