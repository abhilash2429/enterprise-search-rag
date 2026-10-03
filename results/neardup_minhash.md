# Near-duplicate report

511,958 docs, 3,856 pairs with estimated Jaccard >= 0.5

| threshold | pairs | docs in clusters | clusters (size>=2) | max size | p99 size | gold docs with a near-dup |
|---|---|---|---|---|---|---|
| 0.5 | 3,856 | 7,483 | 3,704 | 4 | 3 | 1/722 |
| 0.6 | 3,760 | 7,308 | 3,619 | 4 | 3 | 1/722 |
| 0.7 | 3,639 | 7,097 | 3,518 | 4 | 3 | 1/722 |
| 0.8 | 3,558 | 6,951 | 3,448 | 3 | 3 | 1/722 |
| 0.9 | 3,413 | 6,673 | 3,311 | 3 | 3 | 1/722 |
| 0.95 | 3,180 | 6,227 | 3,091 | 3 | 3 | 1/722 |

Settings: 5-word shingles over the Lucene-standard tokens, 128 permutations, datasketch LSH, pairs kept down to estimated Jaccard 0.5.

## Known versions are reworded, not copied

Exact shingle Jaccard on pairs that are known versions of the same thing:

| Pair set | n | Jaccard |
|---|---|---|
| Same doc_id, two different documents in the parquet | 4 | 0.01, 0.06, 0.05, 0.16 |
| Gold docs of conflicting_info questions (newer supersedes older) | 20 | mean 0.16, max 1.00, only 1 >= 0.5 (an exact copy) |

Nearly all lexical near-duplicate pairs are Slack channel dumps sharing boilerplate. The corpus is LLM-generated, so versions of a
document were written independently and share meaning, not wording. MinHash on text does not find them, which is the job
PROJECT.md gives near-duplicate clustering.
