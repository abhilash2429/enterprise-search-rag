# BM25 baseline (paper reproduction)

OpenSearch 2.19.1, harness index settings and query, top-10 full documents, harness ANSWER_GEN_PROMPT.
Generator and judge: openai.gpt-oss-120b on Bedrock, reasoning effort medium, metrics_based_eval --no-correction. All 500 questions.
Paper columns are BM25 + GPT-5.4 (generator and judge), so judge-based columns are not directly comparable; recall is.

bm25_opensearch: $2.30 total, $0.0046/question

| run             | type                     |   n |   correct |   complete |   overall |   recall |   paper_correct |   paper_recall |
|:----------------|:-------------------------|----:|----------:|-----------:|----------:|---------:|----------------:|---------------:|
| bm25_opensearch | basic                    | 175 |      77.1 |       54.2 |      52.6 |     77.7 |            79.4 |           77.7 |
| bm25_opensearch | semantic                 | 125 |      52.0 |       37.0 |      31.2 |     43.2 |            44.8 |           43.2 |
| bm25_opensearch | intra_document_reasoning |  40 |      82.5 |       63.4 |      58.6 |     90.0 |            85.0 |           90.0 |
| bm25_opensearch | project_related          |  40 |      55.0 |       59.4 |      37.3 |     65.5 |            60.0 |           65.5 |
| bm25_opensearch | constrained              |  30 |      66.7 |       83.4 |      62.1 |     85.0 |            76.7 |           85.0 |
| bm25_opensearch | conflicting_info         |  20 |      90.0 |       67.3 |      64.5 |     82.5 |            90.0 |           82.5 |
| bm25_opensearch | completeness             |  20 |      50.0 |       30.4 |      22.5 |     46.5 |            40.0 |           46.5 |
| bm25_opensearch | miscellaneous            |  20 |      90.0 |       61.8 |      61.8 |     90.0 |            85.0 |           90.0 |
| bm25_opensearch | high_level               |  10 |      50.0 |       53.2 |      44.2 |      0.0 |            50.0 |          nan   |
| bm25_opensearch | info_not_found           |  20 |      70.0 |       70.0 |      70.0 |      0.0 |           100.0 |          nan   |
| bm25_opensearch | overall                  | 500 |      68.0 |       53.3 |      47.2 |     68.4 |            68.8 |           68.4 |
