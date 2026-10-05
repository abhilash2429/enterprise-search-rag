# Demo UI

Next.js (App Router, TypeScript strict, Tailwind) frontend for the demo API. The contract is
[docs/demo-api.md](../docs/demo-api.md); [fixtures/](fixtures/) holds real recorded runs of it, so the UI runs without the
backend, its GPU or its indexes.

Right now the page is bare: it lists the demo questions, and clicking one prints the raw event stream as it arrives.

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

The tests run the client against the mock route handlers in process (no server, no browser): every recorded run replays
through `ask` and each event parses into its type and equals the recorded payload, events follow the contract order,
replay pacing follows `t / NEXT_PUBLIC_MOCK_SPEED`, and every reranked hit's document is served. Synthetic runs built
from the contract cover what no fixture does: a refusal (`generate.abstained: true`, `verify.skipped: "abstained"`,
`confidence: null`), a run without router and verifier, an unparsed verifier reply, the server's `error` event, dropped
and refused streams, the `/health` pre-check, contract drift and abort.
