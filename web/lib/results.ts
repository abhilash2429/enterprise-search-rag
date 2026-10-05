// Results tables from the repository README.md, hardcoded for the /benchmark page.
//
// Every cell is the README's text exactly, kept as a string so nothing is re-rounded or reformatted. Each table names
// the README section it comes from; test/results.test.ts re-parses README.md and fails if any cell, header or row here
// differs from it.

export type ResultTable<K extends string> = {
  id: string
  title: string
  /** README heading path the table sits under. */
  section: string
  /** The README sentence that introduces the table. */
  caption: string
  columns: { key: K; header: string }[]
  rows: Record<K, string>[]
  /** Index of the headline-config row, if the table has one. */
  headline: number | null
}

// README.md, "Results > Held-out test (run once, config locked on dev)"
export const heldOut: ResultTable<"system" | "recall10" | "correct" | "complete" | "combined" | "vsBm25" | "genCost"> = {
  id: "heldOut",
  title: "Held-out test",
  section: "Results > Held-out test (run once, config locked on dev)",
  caption: "350 test questions, 3 generation seeds. Combined = the harness score: completeness if the judge marks the answer correct, else 0, averaged per question over seeds. CIs are bootstrap over questions; deltas are paired.",
  columns: [
    { key: "system", header: "System" },
    { key: "recall10", header: "Recall@10" },
    { key: "correct", header: "Correct" },
    { key: "complete", header: "Complete" },
    { key: "combined", header: "Combined" },
    { key: "vsBm25", header: "vs BM25" },
    { key: "genCost", header: "Gen $/q" },
  ],
  rows: [
    { system: "BM25 (paper config) + harness prompt", recall10: "70.2", correct: "70.4", complete: "53.0", combined: "47.7 [43.5, 52.0]", vsBm25: "", genCost: "0.0028" },
    { system: "Hybrid + rerank + cited answerer", recall10: "81.6", correct: "78.5", complete: "74.2", combined: "67.9 [63.8, 71.8]", vsBm25: "+20.2 [+15.8, +24.4]", genCost: "0.0028" },
    { system: "+ source router (headline)", recall10: "82.2", correct: "78.6", complete: "74.5", combined: "67.9 [63.8, 71.8]", vsBm25: "+20.1 [+15.7, +24.5]", genCost: "0.0028" },
  ],
  headline: 2,
}

// README.md, "Results > Held-out test (run once, config locked on dev)"
export const recallByType: ResultTable<"type" | "bm25" | "headline"> = {
  id: "recallByType",
  title: "Recall@10 by question type",
  section: "Results > Held-out test (run once, config locked on dev)",
  caption: "Recall@10 by question type on test (questions with gold docs, n=329):",
  columns: [
    { key: "type", header: "Type" },
    { key: "bm25", header: "BM25" },
    { key: "headline", header: "Headline" },
  ],
  rows: [
    { type: "basic", bm25: "77.9", headline: "90.2" },
    { type: "semantic", bm25: "51.1", headline: "68.2" },
    { type: "completeness", bm25: "43.6", headline: "51.7" },
    { type: "project_related", bm25: "67.1", headline: "70.4" },
    { type: "conflicting_info", bm25: "78.6", headline: "89.3" },
    { type: "constrained", bm25: "85.7", headline: "95.2" },
    { type: "intra_document_reasoning", bm25: "89.3", headline: "96.4" },
    { type: "miscellaneous", bm25: "85.7", headline: "100.0" },
  ],
  headline: null,
}

// README.md, "Results > Dev ablations"
export const devRetrieval: ResultTable<"stage" | "recall10" | "note"> = {
  id: "devRetrieval",
  title: "Dev ablations: retrieval",
  section: "Results > Dev ablations",
  caption: "Retrieval, recall@10:",
  columns: [
    { key: "stage", header: "Stage" },
    { key: "recall10", header: "Recall@10" },
    { key: "note", header: "Note" },
  ],
  rows: [
    { stage: "BM25, own implementation", recall10: "63.6", note: "OpenSearch with the paper config: 64.3" },
    { stage: "Dense, Qwen3-Embedding-0.6B, 512-token chunks, MaxP", recall10: "48.4", note: "" },
    { stage: "Hybrid, RRF k=60 over BM25 + dense top-100", recall10: "67.6", note: "k in {10, 20, 60, 100} swept, 60 best" },
    { stage: "+ rerank top-100 (Qwen3-Reranker-0.6B)", recall10: "74.5", note: "depth 20: 74.1, depth 50: 75.3" },
    { stage: "+ source router (headline)", recall10: "77.6", note: "+3.1 [+0.8, +6.2] over no router" },
    { stage: "Rerank, model-card instruction instead of custom", recall10: "75.4", note: "+0.9 [-3.3, +5.2], kept custom" },
    { stage: "RRF(rerank, hybrid) instead of pure rerank", recall10: "77.7", note: "end to end +0.1, kept pure rerank" },
  ],
  headline: 4,
}

// README.md, "Results > Dev ablations"
export const devDense: ResultTable<"index" | "hybridRecall10" | "size"> = {
  id: "devDense",
  title: "Dev ablations: dense index variants",
  section: "Results > Dev ablations",
  caption: "Dense index variants (hybrid recall@10, everything else fixed):",
  columns: [
    { key: "index", header: "Dense index" },
    { key: "hybridRecall10", header: "Hybrid recall@10" },
    { key: "size", header: "Size" },
  ],
  rows: [
    { index: "fp16, exact (headline)", hybridRecall10: "67.6", size: "3.15 GB" },
    { index: "int8 scalar", hybridRecall10: "67.6", size: "1.58 GB" },
    { index: "binary, Hamming", hybridRecall10: "63.2", size: "197 MB" },
    { index: "binary + fp16 rescore of a 4x shortlist", hybridRecall10: "67.5", size: "197 MB in RAM, fp16 on disk" },
    { index: "whole-document embedding (first 8192 tokens)", hybridRecall10: "67.1", size: "1.05 GB" },
  ],
  headline: 0,
}

// README.md, "Results > Dev ablations"
export const devEndToEnd: ResultTable<"system" | "correct" | "combined" | "vsBm25" | "genCost"> = {
  id: "devEndToEnd",
  title: "Dev ablations: end to end",
  section: "Results > Dev ablations",
  caption: "End to end, 3 seeds unless noted:",
  columns: [
    { key: "system", header: "System" },
    { key: "correct", header: "Correct" },
    { key: "combined", header: "Combined" },
    { key: "vsBm25", header: "vs BM25" },
    { key: "genCost", header: "Gen $/q" },
  ],
  rows: [
    { system: "BM25 + harness prompt", correct: "62.7", combined: "42.2 [35.5, 48.6]", vsBm25: "", genCost: "0.0029" },
    { system: "Rerank + harness prompt", correct: "68.7", combined: "45.5 [39.0, 51.9]", vsBm25: "+3.3 [-1.8, +8.3]", genCost: "0.0027" },
    { system: "Rerank + cited answerer", correct: "70.2", combined: "57.9 [51.2, 64.4]", vsBm25: "+15.7 [+9.1, +22.3]", genCost: "0.0028" },
    { system: "Rerank + cited answerer, 5 docs of context", correct: "65.3", combined: "55.0 [48.0, 61.9]", vsBm25: "+12.8 [+6.2, +19.6]", genCost: "0.0016" },
    { system: "Rerank + cited answerer, 20 docs of context", correct: "70.4", combined: "59.3 [52.8, 65.9]", vsBm25: "+17.2 [+10.6, +23.9]", genCost: "0.0052" },
    { system: "Router + rerank + cited answerer (headline)", correct: "73.8", combined: "62.2 [55.8, 68.7]", vsBm25: "+20.1 [+13.6, +26.8]", genCost: "0.0028" },
    { system: "gpt-6-luna file agent, 1 seed", correct: "70.7", combined: "58.5 [51.3, 65.8]", vsBm25: "+16.4 [+9.0, +23.9]", genCost: "0.0214" },
  ],
  headline: 5,
}

// README.md, "Results > Serving latency"
export const latency: ResultTable<"stage" | "p50" | "p90" | "where"> = {
  id: "latency",
  title: "Serving latency",
  section: "Results > Serving latency",
  caption: "Online pipeline (`entsearch-mcp`, headline config) on a laptop: RTX 3050 6 GB, i5-13450HX, 16 GB RAM. 50 dev questions.",
  columns: [
    { key: "stage", header: "Stage" },
    { key: "p50", header: "p50" },
    { key: "p90", header: "p90" },
    { key: "where", header: "Where it runs" },
  ],
  rows: [
    { stage: "Route", p50: "0.7 s", p90: "1.3 s", where: "gpt-oss-120b on Bedrock, low effort" },
    { stage: "Retrieve, fuse, fetch", p50: "8.0 s", p90: "10.4 s", where: "BM25 in process; dense exact scan of the 3.15 GB fp16 index on CPU, memory-mapped" },
    { stage: "Rerank 100 docs", p50: "34.2 s", p90: "41.8 s", where: "Qwen3-Reranker-0.6B fp16 on the GPU, up to 4096 tokens per doc" },
    { stage: "Generate", p50: "3.9 s", p90: "12 s", where: "gpt-oss-120b on Bedrock" },
  ],
  headline: null,
}

// README.md, "Findings > When the system is wrong, it rarely says so, and a verifier flag helps only partly"
export const confidenceFlag: ResultTable<"split" | "answers" | "wrong" | "flagged" | "wrongCaught" | "correctFlagged" | "correctIfNotFlagged" | "correctIfFlagged"> = {
  id: "confidenceFlag",
  title: "Confidence flag",
  section: "Findings > When the system is wrong, it rarely says so, and a verifier flag helps only partly",
  caption: "The threshold was picked on dev by max F1; the test row is one pass with everything locked, on seed-0 answers.",
  columns: [
    { key: "split", header: "Split" },
    { key: "answers", header: "Answers" },
    { key: "wrong", header: "Wrong" },
    { key: "flagged", header: "Flagged" },
    { key: "wrongCaught", header: "Wrong answers caught" },
    { key: "correctFlagged", header: "Correct answers flagged" },
    { key: "correctIfNotFlagged", header: "Correct if not flagged" },
    { key: "correctIfFlagged", header: "Correct if flagged" },
  ],
  rows: [
    { split: "Dev", answers: "145", wrong: "27.6%", flagged: "58", wrongCaught: "77.5% [65.0, 90.0]", correctFlagged: "25.7% [17.1, 34.3]", correctIfNotFlagged: "89.7% [82.8, 95.4]", correctIfFlagged: "46.6%" },
    { split: "Test", answers: "335", wrong: "20.0%", flagged: "140", wrongCaught: "71.6% [59.7, 82.1]", correctFlagged: "34.3% [28.7, 40.3]", correctIfNotFlagged: "90.3% [86.2, 94.4]", correctIfFlagged: "65.7% [57.9, 73.6]" },
  ],
  headline: null,
}

export type ResultGroup = { title: string; note: string | null; tables: ResultTable<string>[] }

/** In README order, grouped as on the /benchmark page. */
export const RESULT_GROUPS: ResultGroup[] = [
  { title: "Held-out test", note: "Run once, config locked on dev.", tables: [heldOut, recallByType] },
  { title: "Dev ablations", note: "150 dev questions (141 with gold docs for recall). Every row was decided before test was touched.", tables: [devRetrieval, devDense, devEndToEnd] },
  { title: "Serving latency", note: null, tables: [latency] },
  { title: "Confidence flag", note: null, tables: [confidenceFlag] },
]

export const ALL_TABLES = RESULT_GROUPS.flatMap((g) => g.tables)
