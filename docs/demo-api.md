# Demo API contract

The demo UI talks to `entsearch-demo` (`src/entsearch/serve/demo.py`), which serves the headline pipeline from
`src/entsearch/serve/pipeline.py`: source router, hybrid retrieval (BM25 + dense, each also restricted to the routed
sources), RRF fusion to 100 candidates, Qwen3-Reranker to the top 10, cited answer from gpt-oss-120b, and a confidence
flag from a gpt-6-luna verifier.

The backend runs only on the author's laptop (GPU, 7 GB of indexes, Bedrock and Azure credentials). The frontend is built
against this contract and the recorded runs in `web/fixtures/`, which are real outputs of this API.

Base URL in live mode: `http://localhost:8000/api`. CORS allows `http://localhost:3000` and `http://127.0.0.1:3000`, GET
only. All bodies are JSON. Errors are `{"error": string}` with a non-200 status.

## Endpoints

### `GET /health`

```ts
type Health = {
  status: "loading" | "ready" | "error";  // indexes load in about a minute after start
  busy: boolean;                          // a question is running; /ask would return 409
  config: Config;
  error: string | null;                   // set when status is "error"
};
type Config = { router: boolean; rerank: boolean; dense: "fp16" | "binary" | "none"; verify: boolean };
```

### `GET /questions`

Curated benchmark questions for the demo, with the benchmark's gold answer and gold documents.

```ts
type Question = {
  question_id: string;            // "qst_0147"
  question: string;
  question_type: string;          // basic | semantic | conflicting_info | project_related | info_not_found | ...
  gold_answer: string;
  expected_doc_ids: string[];     // gold documents; may be empty (info_not_found)
};
// response: Question[]
```

### `GET /documents/{doc_id}`

```ts
type Document = { doc_id: string; source: Source; title: string; content: string };
```

404 for an unknown id, 503 while loading. `content` is plain text, sometimes long (Slack and email threads run to tens of
thousands of characters), with newlines.

`Source` is one of the 9 corpus sources: `slack`, `gmail`, `google_drive`, `confluence`, `jira`, `linear`, `github`,
`hubspot`, `fireflies`.

### `GET /ask?q=<question>` (server-sent events)

Runs one question and streams one event per completed stage. Status codes before the stream starts: 400 (missing `q`),
503 (loading or load error), 409 (another question is running; check `busy` in `/health` first, since `EventSource`
cannot read error bodies). One question runs at a time.

Each SSE message has `event: <name>` and `data: <JSON>`. Events arrive in this order; a stage that is not in
`start.stages` is never sent. The stage after the last completed one is the one currently running, so the UI can show it
as in progress.

| Event | When | Stage time in the recorded runs (min / median / max) |
|---|---|---|
| `start` | immediately | |
| `route` | router picked sources | 1.2 / 1.8 / 2.4 s |
| `retrieve` | BM25 and dense lists ready | 3.5 / 4.3 / 7.7 s |
| `fuse` | RRF top 100 with titles | 0.1 / 0.1 / 0.4 s |
| `rerank` | top 10 reranked | 8.9 / 33.8 / 43.2 s |
| `generate` | cited answer | 2.0 / 6.3 / 21.2 s |
| `verify` | confidence flag | 4.8 / 8.5 / 23.5 s |
| `done` | end of stream; whole question 23.5 / 54.7 / 80.5 s | |
| `error` | instead of the remaining events, then the stream ends | |

Payloads:

```ts
type Start = { question: string; config: Config; stages: Stage[] };
type Stage = "route" | "retrieve" | "fuse" | "rerank" | "generate" | "verify";

type Route = { sources: Source[]; seconds: number };   // 1 to 3 sources; [] if the router named none

type ListName = "bm25" | "dense" | "bm25_routed" | "dense_routed";
type Retrieve = {
  lists: Record<ListName, string[] | null>;  // doc ids, best first, up to 100 each; routed lists null when no route
  seconds: number;
};

type Fuse = {
  candidates: {
    doc_id: string; source: Source; title: string;
    rrf_score: number;                        // sum of 1/(60 + rank) over the lists that found it
    ranks: Record<ListName, number | null>;   // 1-based rank in each list, null if that list did not return it
  }[];                                        // 100 items, fused order (best first)
  seconds: number;                            // includes fetching the 100 documents' titles
};

type Rerank = {
  hits: {
    rank: number;           // 1..10, reranked order
    doc_id: string; source: Source; title: string;
    snippet: string;        // first 300 characters of the content
    rerank_score: number;   // 0..1, reranker's relevance probability
    fused_rank: number;     // 1..100, where RRF had it before reranking
  }[];                      // the 10 documents the answerer sees
  order: { doc_id: string; rerank_score: number }[];  // all 100 candidates in reranked order
  seconds: number;
};

type Generate = {
  answer: string;           // plain text with citation markers [n] (also [n][m] or [n, m]); n indexes `context`
  abstained: boolean;       // answer is exactly the fixed refusal sentence: the documents do not answer it
  partial: boolean;         // answer contains a sentence starting "Not covered by the documents:"
  citations: { n: number; doc_id: string; source: Source; title: string }[];  // docs the answer cites, first-cited order
  context: { n: number; doc_id: string; source: Source; title: string }[];    // all 10 docs, n = 1..10 = rerank rank
  version_pairs: { a: number; b: number; cosine: number }[];  // context docs that look like versions of each other
  seconds: number;
  cost_usd: number;
};

type Verify = {
  confidence: Confidence | null;   // null when skipped
  skipped: "abstained" | null;     // refusals are not verified
  seconds: number;
  cost_usd: number;
};
type Confidence = {
  flagged: boolean | null;         // true: double-check the cited documents. null: verifier reply did not parse (see error)
  verdict?: "supported" | "partially_supported" | "unsupported";
  addresses_question?: boolean;
  low_retrieval_score?: boolean;   // reranker top score below threshold (also flags)
  rerank_top: number | null;
  threshold?: number;              // 0.9489
  claims?: { claim: string; cited: number[]; verdict: "supported" | "partially_supported" | "unsupported" | "contradicted" }[];
  reasoning?: string;              // two or three sentences from the verifier
  error?: string;
};

type Done = {
  seconds: { route?: number; retrieve: number; fetch: number; rerank?: number; generate: number; verify?: number; total: number };
  cost_usd: number;                // whole question, USD
};
type ErrorEvent = { message: string };
```

The flag never changes the answer. Measured on held-out test answers: unflagged answers are right about 90% of the time,
flagged ones about 66%; two of three flags are false alarms (README, Findings).

## Fixtures (`web/fixtures/`)

Recorded from this API with fresh LLM calls, so timings are real cold timings on the author's laptop.

- `health.json`: a `Health` in the ready state.
- `questions.json`: the `Question[]` that `/questions` returns. Every one has a recorded run.
- No recorded run is a refusal (`abstained: true`): on fresh samples the answerer answered all three info-not-found
  questions tried, and the verifier flagged each answer. The refusal path (and `verify.skipped: "abstained"`) still has to
  work; test it with a synthetic event built from this contract.
- `ask/<question_id>.jsonl`: one line per event, `{"t": number, "event": string, "data": object}`, where `t` is seconds
  since the request started. Replaying with these offsets reproduces the real pacing.
- `documents/<doc_id>.json`: a `Document` for every doc in any recorded `rerank.hits`, so the drawer works offline.
  Candidates that only appear in `fuse` have titles but no document file.
