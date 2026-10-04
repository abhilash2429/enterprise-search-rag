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
| 2026-10-03 | Near-dup clustering: MinHash on text (word shingles + LSH); threshold picked from the measured distribution and inspected samples | Matches the stated job (copies differing in small facts); cheap, deterministic, explainable. |
| 2026-10-04 | Near-dup method switched to embedding similarity over doc vectors | MinHash found 1/722 gold docs with a near-dup; known version pairs have Jaccard 0.01-0.16 (results/neardup_minhash.md). Threshold tuned on dev-question pairs only. |
| 2026-10-04 | Dense baseline: recall@10 52.5 on all 500 (paper vector baseline 46.0) | Paper-faithful config; untuned, so run on all 500 like BM25. |
| 2026-10-04 | Embedding near-dups: mean pooling, cosine 0.88, flagged pairwise among the retrieved top-10 at query time; no corpus-wide clustering | 0.88 picked on dev labeled pairs (83% of versions, 5% of hard negatives). Union-find over corpus kNN chains topical neighbours into a 146K-doc cluster (results/neardup_embed.md). |
| 2026-10-04 | Own BM25 (bm25.py) replaces OpenSearch for runs: recall@10 68.3 vs 68.4 on all 500 | Same Lucene formula over exact doc lengths; residual gap is likely Lucene's lossy length norms. In-process, ~90 ms/query, no container. |
| 2026-10-04 | Hybrid: RRF over BM25 top-100 docs and dense top-100 docs (MaxP over top-4000 chunks), k=60 headline | Dev recall@10 67.6 vs BM25 63.6 (paired delta +4.0, 95% CI [-2.0, +9.9], n=141). k fixed at the paper value, sweep {10, 20, 60, 100} is an ablation (k=60 also best on dev). Fused top-100 holds 83.3% of dev gold, the reranker's ceiling. |
