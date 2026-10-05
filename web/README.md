# Demo UI

Next.js (App Router, TypeScript strict, Tailwind) frontend for the demo API. The contract is
[docs/demo-api.md](../docs/demo-api.md); [fixtures/](fixtures/) holds real recorded runs of it, so the UI runs without the
backend, its GPU or its indexes.

## The screen

The UI follows the M-Rag-B frontend (github.com/abhilash2429/M-Rag-B): same layout, palette, Geist type and vendored
prompt-kit / shadcn primitives in `components/ui/` (copied unchanged).

- Left: the benchmark questions from `/questions`, and "New session" to clear the conversation.
- Middle: a chat thread. Each answer opens with the pipeline steps. Collapsed, one line names the step in progress
  and changes as each step starts ("Reranking 100 candidates"), then reads "Ran 6 steps"; the total time and cost sit
  right-aligned on that line. Clicking it expands the steps into a vertical list that grows as each step starts
  (Route, Retrieve, Fuse, Rerank, Generate, Verify), each with what it produced (the routed sources, the candidate
  count, the citations, the verifier's result) and its real seconds, or a live timer while it runs. The setting holds
  for later questions. Then the answer as markdown with citation chips. `[n]`, `[n][m]`, `[n, m]` and the backend's `【n】` form all become chips.
  Banners for a refusal and a partial answer; the "Not covered by the documents:" sentence is set apart; version pairs
  are noted; the confidence check sits under the answer ("Check the cited documents" with the verdict, reasoning and
  unsupported claims when flagged, a quiet "Verified" line otherwise, and the low-retrieval note). "Why this answer"
  expands the per-stage details.
- Right: the evidence rail with the 10 reranked documents (rank, source with its colour, title, snippet, score, rank
  before reranking, cited marker). While reranking runs it shows the top of the fused list. A citation chip highlights
  and scrolls to its document; "Open document" shows the full text from `/documents/{id}`.

- Retrieval tab (next to Evidence, or press R): all 100 fused candidates with their 1-based rank in BM25, dense,
  BM25 routed and dense routed (blank when that list missed it; filled chips for the full lists, outlined for the
  routed ones), the RRF score, and the reranked position for the top 10. Rows found by only one retriever (BM25 or
  dense, counting its routed list as the same retriever) are tinted and labelled, with counts at the top.
- When rerank arrives, the evidence rail first shows the fused top 10, marks each kept (with its new rank) or dropped,
  then moves the documents to their reranked positions; documents promoted from deeper in the fused list slide in.
- "Show gold answer" under answers to questions from `/questions`: the benchmark's gold answer and, for each gold
  document, its rank in the top 10 or that it was missed (with its fused rank when it was a candidate).
- Errors say what happened and offer Retry: indexes loading (polls `/health` every 2 s and reruns the question when
  ready), another question running, an `error` event, a dropped connection, a response that breaks the contract.

`/benchmark` shows the README's results tables (held-out test, recall@10 by type, dev ablations, serving latency,
confidence flag) from `lib/results.ts`, which holds every cell as the README's exact text; `test/results.test.ts`
re-parses `README.md` and fails on any difference.

`/raw` keeps the bare page that prints the raw event stream.

### Recording

| Key | Does |
|---|---|
| `/` | Focus the question input |
| Enter | Ask |
| `1`-`8` | Ask demo question 1-8 |
| `R` | Toggle the Retrieval tab |
| Esc | Close the document viewer |

`?record=1` hides dev-only UI (mode label, key hints, API details, theme button), makes everything one step larger
and hides the cursor after 2 s without movement. `?theme=dark` or `?theme=light` picks the theme (the button in the
header does too, and remembers it). Both themes pass axe's WCAG 2.1 A/AA rules, contrast included, in the e2e check.

## Setup

Node 20.9 or newer.

```bash
cd web
npm install
```

## Mock mode (default, offline)

```bash
npm run dev                                  # http://localhost:3000
NEXT_PUBLIC_MOCK_SPEED=10 npm run dev        # replay 10x faster
```

With `NEXT_PUBLIC_API_BASE` unset (or set to `/mock-api`), the client talks to route handlers under
[app/mock-api/](app/mock-api/), which serve the fixtures:

| Route | Serves |
|---|---|
| `GET /mock-api/health` | `fixtures/health.json` |
| `GET /mock-api/questions` | `fixtures/questions.json` |
| `GET /mock-api/documents/{doc_id}` | `fixtures/documents/{doc_id}.json`, or 404 `{"error": ...}` |
| `GET /mock-api/ask?q=` | `fixtures/ask/{question_id}.jsonl` replayed as `text/event-stream` |

`/ask` matches `q` to a fixture by question text (trimmed) and sends each event at its recorded `t` divided by
`NEXT_PUBLIC_MOCK_SPEED` (default 1, the real pacing: 23 to 80 s per question). A question with no recorded run streams
a single `error` event saying only the demo questions work offline. A missing `q` is a 400, as in the contract.

Documents exist only for the docs in a recorded `rerank.hits`; candidates that appear only in `fuse` have no file.

## Live mode (backend on the laptop)

Start the backend from the repo root (indexes take about a minute to load; `/health` reports `loading` until then):

```bash
uv sync --extra demo
uv run entsearch-demo                        # http://127.0.0.1:8000/api
```

Then, in `web/`:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000/api npm run dev
```

Open the UI at `http://localhost:3000` or `http://127.0.0.1:3000`: the backend's CORS allows only those two origins.

`NEXT_PUBLIC_*` values are inlined when `npm run dev` or `npm run build` starts, so restart (or rebuild) after changing
them.

## How the client works

[lib/api.ts](lib/api.ts) has every type from the contract copied verbatim, a strict zod schema for each (unknown keys,
unknown enum values and missing fields are errors), and a compile-time check that each schema infers exactly its contract
type. `createClient()` exposes `getHealth`, `getQuestions`, `getDocument` and `ask`.

`ask(question, onEvent, signal)`:

- Checks `/health` first and resolves `rejected` if the backend is loading, failed to load or busy, because
  `EventSource` cannot read the bodies of 409 and 503 responses.
- Opens an `EventSource` on `/ask?q=`, validates each event, and calls `onEvent(event, rawData)`.
- Closes the `EventSource` on `done`, on the server's `error` event, on a dropped connection, on a payload that breaks
  the contract, or on abort. Closing matters: a browser `EventSource` reconnects on its own, which would re-run the
  question (or hit 409).
- The server's `error` event and `EventSource`'s connection error share the name `error`; only the server's is a
  `MessageEvent` with `data`, which is how the client tells them apart.
- Resolves with an `AskOutcome`: `done`, `error`, `rejected`, `disconnected`, `invalid` or `aborted`.

## Scripts

| Script | Does |
|---|---|
| `npm run dev` | Next dev server on port 3000 |
| `npm run build` | Production build (`npm start` serves it) |
| `npm run lint` | ESLint (Next core-web-vitals and TypeScript rules) |
| `npm run typecheck` | `next typegen` then `tsc --noEmit`, including the schema-equals-contract check |
| `npm test` | Vitest |
| `npm run e2e` | Every fixture end to end in headless Chromium against a running app (see below) |
| `npm run e2e:video` | Records one full run in recording mode as a 1920x1080 video |

The tests run the client against the mock route handlers in process (no server, no browser): every recorded run replays
through `ask` and each event parses into its type and equals the recorded payload, events follow the contract order,
replay pacing follows `t / NEXT_PUBLIC_MOCK_SPEED`, and every reranked hit's document is served. Synthetic runs built
from the contract cover what no fixture does: a refusal (`generate.abstained: true`, `verify.skipped: "abstained"`,
`confidence: null`), a run without router and verifier, an unparsed verifier reply, the server's `error` event, dropped
and refused streams, the `/health` pre-check, contract drift and abort.

### End-to-end check

`e2e/run.mjs` drives the built app at 1920x1080: every recorded run (the collapsed step line, steps appearing as they
start with their seconds, routed sources, total and cost on the header row, chips matching `generate.citations`, banners, version pairs, confidence, the rerank reorder, evidence order
and cited marks, chip highlight and scroll, the document viewer, every Retrieval row, the gold answer), a synthetic
refusal, an unknown question, the keyboard shortcuts, the four error states with Retry, recording mode (including no page scroll at 1920x1080 and
1600x900 with an answer on screen, Evidence and Retrieval), the benchmark
page against `README.md`, and axe WCAG 2.1 A/AA scans in both themes. Set the same speed for the
server and the check:

```bash
NEXT_PUBLIC_MOCK_SPEED=10 npm run build && NEXT_PUBLIC_MOCK_SPEED=10 npm start    # shell 1
NEXT_PUBLIC_MOCK_SPEED=10 npm run e2e                                             # shell 2
```

It needs a Chromium for Playwright (`npx playwright install chromium` once, or `PLAYWRIGHT_CHROMIUM_PATH`).
`E2E_SHOTS=<dir>` saves a screenshot of each finished answer.

`e2e/record-video.mjs` (`npm run e2e:video`) records one full run (default `qst_0421`) in recording mode the way a
presenter would drive it: key `2`, the whole pipeline at the server's pacing, a citation, its document, then `R`.
`E2E_VIDEO=<dir>` sets the output folder, `E2E_QUESTION` and `E2E_THEME` the run.
