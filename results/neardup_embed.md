# Embedding near-duplicates

Doc vector = mean of the doc's Qwen3-Embedding-0.6B chunk vectors, L2-normalized. 511,958 docs.

## Threshold from labeled pairs

Positives: gold doc pairs of conflicting_info questions (an older and a newer version of the same fact).
Hard negatives: gold doc pairs of completeness and project_related questions (related but distinct docs).

| split | | 0.80 | 0.85 | 0.88 | 0.90 | 0.92 | 0.94 | 0.96 |
|---|---|---|---|---|---|---|---|---|
| dev (6 pos, 225 neg) | positives >= t | 1.00 | 1.00 | 0.83 | 0.67 | 0.50 | 0.50 | 0.50 |
| | negatives >= t | 0.29 | 0.12 | 0.05 | 0.03 | 0.00 | 0.00 | 0.00 |
| test (13 pos, 531 neg) | positives >= t | 1.00 | 1.00 | 0.92 | 0.69 | 0.46 | 0.38 | 0.23 |
| | negatives >= t | 0.32 | 0.18 | 0.07 | 0.05 | 0.01 | 0.00 | 0.00 |

Mean pooling separates better than first-chunk pooling. Picked 0.88 on dev.

## Corpus-wide clustering does not work

Exact kNN (k=10) over all docs, pairs kept down to cosine 0.80, union-find clusters:

| threshold | pairs | docs in clusters | clusters (size>=2) | max size | p99 size | gold docs with a near-dup |
|---|---|---|---|---|---|---|
| 0.80 | 3,518,403 | 488,240 | 2,356 | 478,270 | 39 | 648/722 |
| 0.85 | 2,070,491 | 391,259 | 12,379 | 333,666 | 51 | 506/722 |
| 0.88 | 928,777 | 257,692 | 21,977 | 146,656 | 58 | 347/722 |
| 0.90 | 398,508 | 153,693 | 23,796 | 31,714 | 51 | 233/722 |
| 0.92 | 127,338 | 70,384 | 17,883 | 4,576 | 31 | 102/722 |
| 0.94 | 26,276 | 26,405 | 9,955 | 267 | 9 | 41/722 |
| 0.96 | 5,732 | 10,756 | 5,215 | 13 | 3 | 16/722 |
| 0.98 | 3,792 | 7,384 | 3,657 | 3 | 3 | 1/722 |

At 0.88 one cluster holds 146,656 docs from all 9 sources. Sampled pairs in the 0.88-0.92 band are same-topic docs, not
versions: two different HubSpot company records at 0.905, separate gmail threads about the same customer demo. Single-linkage
chains these into giant clusters. Clusters only become sane near 0.96, where most real version pairs (mean cosine 0.93) are lost.
A doc-level embedding does not distinguish "same topic" from "same document, revised", and the 6 labeled dev pairs could not show
the corpus-wide base rate.

## Query-time pairwise flagging instead

Among the top-10 retrieved docs, every pair with cosine >= 0.88 is flagged as a likely version pair. No chaining.
Dev questions (n=150):

| run | flagged pairs per top-10 | questions with >= 1 flag | conflicting_info: both versions retrieved / flagged |
|---|---|---|---|
| BM25 (OpenSearch) | 2.07 | 47% | 5 / 4 |
| Dense (Qwen3-0.6B) | 2.17 | 60% | 4 / 3 |

Flags fire on about half of questions, so most are topical neighbours rather than versions. They are candidates for conflict
handling to resolve with dates and content, not a dedupe.
