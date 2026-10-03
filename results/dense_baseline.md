# Dense baseline

Qwen3-Embedding-0.6B, full 1024 dims, paper chunking (512 cl100k tokens, 10% overlap), exact cosine search over 1,538,921
chunks, top-100 chunks collapsed to top-10 docs. Query instruction from the model card. All 500 questions, 470 with gold docs.

Recall@10 (%):

| question type | BM25 (OpenSearch) | Dense (Qwen3-0.6B) |
|---|---|---|
| basic | 77.7 | 66.3 |
| semantic | 43.2 | 20.8 |
| intra_document_reasoning | 90.0 | 62.5 |
| project_related | 65.5 | 48.1 |
| constrained | 85.0 | 83.3 |
| conflicting_info | 82.5 | 60.0 |
| completeness | 46.5 | 28.2 |
| miscellaneous | 90.0 | 90.0 |
| overall | 68.4 | 52.5 |
| dev only | 64.3 | 48.4 |

Paper vector baseline: 46.0 overall. Dense beats BM25 on 37 questions and loses on 134.
