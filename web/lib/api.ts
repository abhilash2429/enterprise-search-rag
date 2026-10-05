// Typed client for the demo API (docs/demo-api.md).
//
// The types below are copied from the contract. Each one has a zod schema that validates payloads at runtime; the
// schemas are strict (unknown keys are rejected) so drift between the backend and the contract shows up as an error
// instead of silently. A compile-time check at the bottom of the schema section fails `npm run typecheck` if a
// schema's inferred type stops matching the copied type exactly.

import { z } from "zod";

// ---------------------------------------------------------------------------------------------------------------------
// Contract types (docs/demo-api.md)

// `Source` is given in prose: one of the 9 corpus sources.
export type Source =
  | "slack"
  | "gmail"
  | "google_drive"
  | "confluence"
  | "jira"
  | "linear"
  | "github"
  | "hubspot"
  | "fireflies";

export type Health = {
  status: "loading" | "ready" | "error";  // indexes load in about a minute after start
  busy: boolean;                          // a question is running; /ask would return 409
  config: Config;
  error: string | null;                   // set when status is "error"
};
export type Config = { router: boolean; rerank: boolean; dense: "fp16" | "binary" | "none"; verify: boolean };

export type Question = {
  question_id: string;            // "qst_0147"
  question: string;
  question_type: string;          // basic | semantic | conflicting_info | project_related | info_not_found | ...
  gold_answer: string;
  expected_doc_ids: string[];     // gold documents; may be empty (info_not_found)
};
// response: Question[]

export type Document = { doc_id: string; source: Source; title: string; content: string };

export type Start = { question: string; config: Config; stages: Stage[] };
export type Stage = "route" | "retrieve" | "fuse" | "rerank" | "generate" | "verify";

export type Route = { sources: Source[]; seconds: number };   // 1 to 3 sources; [] if the router named none

export type ListName = "bm25" | "dense" | "bm25_routed" | "dense_routed";
export type Retrieve = {
  lists: Record<ListName, string[] | null>;  // doc ids, best first, up to 100 each; routed lists null when no route
  seconds: number;
};

export type Fuse = {
  candidates: {
    doc_id: string; source: Source; title: string;
    rrf_score: number;                        // sum of 1/(60 + rank) over the lists that found it
    ranks: Record<ListName, number | null>;   // 1-based rank in each list, null if that list did not return it
  }[];                                        // 100 items, fused order (best first)
  seconds: number;                            // includes fetching the 100 documents' titles
};

export type Rerank = {
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

export type Generate = {
  answer: string;           // plain text with citation markers [n] (also [n][m] or [n, m]); n indexes `context`
  abstained: boolean;       // answer is exactly the fixed refusal sentence: the documents do not answer it
  partial: boolean;         // answer contains a sentence starting "Not covered by the documents:"
  citations: { n: number; doc_id: string; source: Source; title: string }[];  // docs the answer cites, first-cited order
  context: { n: number; doc_id: string; source: Source; title: string }[];    // all 10 docs, n = 1..10 = rerank rank
  version_pairs: { a: number; b: number; cosine: number }[];  // context docs that look like versions of each other
  seconds: number;
  cost_usd: number;
};

export type Verify = {
  confidence: Confidence | null;   // null when skipped
  skipped: "abstained" | null;     // refusals are not verified
  seconds: number;
  cost_usd: number;
};
export type Confidence = {
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

export type Done = {
  seconds: { route?: number; retrieve: number; fetch: number; rerank?: number; generate: number; verify?: number; total: number };
  cost_usd: number;                // whole question, USD
};
export type ErrorEvent = { message: string };

// Fixture note: the contract says `Generate.answer` uses [n] markers in plain text, but recorded answers also use
// fullwidth brackets (【2】 in qst_0180, 【1】 in qst_0459) and markdown bold (**...**). Nothing here parses markers yet.

// Not in the contract as a type, but stated in prose: errors are `{"error": string}` with a non-200 status.
export type ErrorBody = { error: string };

// ---------------------------------------------------------------------------------------------------------------------
// Runtime schemas

const SourceSchema = z.enum([
  "slack",
  "gmail",
  "google_drive",
  "confluence",
  "jira",
  "linear",
  "github",
  "hubspot",
  "fireflies",
]);
const StageSchema = z.enum(["route", "retrieve", "fuse", "rerank", "generate", "verify"]);

const ConfigSchema = z.strictObject({
  router: z.boolean(),
  rerank: z.boolean(),
  dense: z.enum(["fp16", "binary", "none"]),
  verify: z.boolean(),
});

export const HealthSchema = z.strictObject({
  status: z.enum(["loading", "ready", "error"]),
  busy: z.boolean(),
  config: ConfigSchema,
  error: z.string().nullable(),
});

export const QuestionSchema = z.strictObject({
  question_id: z.string(),
  question: z.string(),
  question_type: z.string(),
  gold_answer: z.string(),
  expected_doc_ids: z.array(z.string()),
});
export const QuestionsSchema = z.array(QuestionSchema);

export const DocumentSchema = z.strictObject({
  doc_id: z.string(),
  source: SourceSchema,
  title: z.string(),
  content: z.string(),
});

export const ErrorBodySchema = z.strictObject({ error: z.string() });

const perList = <T extends z.ZodType>(value: T) =>
  z.strictObject({ bm25: value, dense: value, bm25_routed: value, dense_routed: value });

const DocRefSchema = z.strictObject({ n: z.number(), doc_id: z.string(), source: SourceSchema, title: z.string() });
const VerdictSchema = z.enum(["supported", "partially_supported", "unsupported"]);

const StartSchema = z.strictObject({ question: z.string(), config: ConfigSchema, stages: z.array(StageSchema) });
const RouteSchema = z.strictObject({ sources: z.array(SourceSchema), seconds: z.number() });
const RetrieveSchema = z.strictObject({ lists: perList(z.array(z.string()).nullable()), seconds: z.number() });
const FuseSchema = z.strictObject({
  candidates: z.array(
    z.strictObject({
      doc_id: z.string(),
      source: SourceSchema,
      title: z.string(),
      rrf_score: z.number(),
      ranks: perList(z.number().nullable()),
    }),
  ),
  seconds: z.number(),
});
const RerankSchema = z.strictObject({
  hits: z.array(
    z.strictObject({
      rank: z.number(),
      doc_id: z.string(),
      source: SourceSchema,
      title: z.string(),
      snippet: z.string(),
      rerank_score: z.number(),
      fused_rank: z.number(),
    }),
  ),
  order: z.array(z.strictObject({ doc_id: z.string(), rerank_score: z.number() })),
  seconds: z.number(),
});
const GenerateSchema = z.strictObject({
  answer: z.string(),
  abstained: z.boolean(),
  partial: z.boolean(),
  citations: z.array(DocRefSchema),
  context: z.array(DocRefSchema),
  version_pairs: z.array(z.strictObject({ a: z.number(), b: z.number(), cosine: z.number() })),
  seconds: z.number(),
  cost_usd: z.number(),
});
const ConfidenceSchema = z.strictObject({
  flagged: z.boolean().nullable(),
  verdict: VerdictSchema.optional(),
  addresses_question: z.boolean().optional(),
  low_retrieval_score: z.boolean().optional(),
  rerank_top: z.number().nullable(),
  threshold: z.number().optional(),
  claims: z
    .array(
      z.strictObject({
        claim: z.string(),
        cited: z.array(z.number()),
        verdict: z.enum(["supported", "partially_supported", "unsupported", "contradicted"]),
      }),
    )
    .optional(),
  reasoning: z.string().optional(),
  error: z.string().optional(),
});
const VerifySchema = z.strictObject({
  confidence: ConfidenceSchema.nullable(),
  skipped: z.literal("abstained").nullable(),
  seconds: z.number(),
  cost_usd: z.number(),
});
const DoneSchema = z.strictObject({
  seconds: z.strictObject({
    route: z.number().optional(),
    retrieve: z.number(),
    fetch: z.number(),
    rerank: z.number().optional(),
    generate: z.number(),
    verify: z.number().optional(),
    total: z.number(),
  }),
  cost_usd: z.number(),
});
const ErrorEventSchema = z.strictObject({ message: z.string() });

export type AskEventMap = {
  start: Start;
  route: Route;
  retrieve: Retrieve;
  fuse: Fuse;
  rerank: Rerank;
  generate: Generate;
  verify: Verify;
  done: Done;
  error: ErrorEvent;
};
export type AskEventName = keyof AskEventMap;
export type AskEvent = { [K in AskEventName]: { event: K; data: AskEventMap[K] } }[AskEventName];

const EVENT_SCHEMAS = {
  start: StartSchema,
  route: RouteSchema,
  retrieve: RetrieveSchema,
  fuse: FuseSchema,
  rerank: RerankSchema,
  generate: GenerateSchema,
  verify: VerifySchema,
  done: DoneSchema,
  error: ErrorEventSchema,
} as const;

/** Every SSE event name in the contract, in stream order (`error` replaces the remaining events). */
export const ASK_EVENTS = Object.keys(EVENT_SCHEMAS) as AskEventName[];

// Compile-time proof that every schema infers exactly the contract type above.
type Equals<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
type Assert<T extends true> = T;
export type SchemaMatchesContract = [
  Assert<Equals<z.infer<typeof HealthSchema>, Health>>,
  Assert<Equals<z.infer<typeof QuestionSchema>, Question>>,
  Assert<Equals<z.infer<typeof DocumentSchema>, Document>>,
  Assert<Equals<z.infer<typeof ErrorBodySchema>, ErrorBody>>,
  Assert<Equals<z.infer<typeof StageSchema>, Stage>>,
  Assert<Equals<{ [K in AskEventName]: z.infer<(typeof EVENT_SCHEMAS)[K]> }, AskEventMap>>,
];

/** Raised when a payload does not match the contract. */
export class ContractError extends Error {
  constructor(what: string, issues: z.core.$ZodIssue[]) {
    super(`${what} does not match docs/demo-api.md: ${z.prettifyError(new z.ZodError(issues))}`);
    this.name = "ContractError";
  }
}

function parseWith<T>(schema: z.ZodType<T>, value: unknown, what: string): T {
  const result = schema.safeParse(value);
  if (!result.success) throw new ContractError(what, result.error.issues);
  return result.data;
}

export function isAskEventName(name: string): name is AskEventName {
  return Object.hasOwn(EVENT_SCHEMAS, name);
}

/** Validates one SSE event's JSON payload against its contract type. */
export function parseAskEvent(event: string, data: unknown): AskEvent {
  if (!isAskEventName(event)) throw new ContractError(`event "${event}"`, []);
  const schema: z.ZodType<AskEventMap[typeof event]> = EVENT_SCHEMAS[event];
  return { event, data: parseWith(schema, data, `"${event}" event`) } as AskEvent;
}

export const parseHealth = (value: unknown): Health => parseWith(HealthSchema, value, "/health");
export const parseQuestions = (value: unknown): Question[] => parseWith(QuestionsSchema, value, "/questions");
export const parseDocument = (value: unknown): Document => parseWith(DocumentSchema, value, "/documents/{doc_id}");

// ---------------------------------------------------------------------------------------------------------------------
// Client

export const MOCK_API_BASE = "/mock-api";

/** NEXT_PUBLIC_API_BASE, inlined at build time. The default "/mock-api" is mock mode (recorded fixtures). */
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || MOCK_API_BASE).replace(/\/+$/, "");
export const isMockMode = (base: string = API_BASE): boolean => base === MOCK_API_BASE;

/** A non-200 response; `message` is the body's `error` when it has one. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** The subset of EventSource the client uses, so tests can substitute their own. */
export type EventSourceLike = {
  addEventListener(type: string, listener: (event: Event) => void): void;
  close(): void;
};
export type EventSourceConstructor = new (url: string) => EventSourceLike;

export type ClientOptions = {
  base?: string;
  fetch?: typeof fetch;
  EventSource?: EventSourceConstructor;
};

/** How an /ask call ended. The promise from `ask` always resolves with one of these. */
export type AskOutcome =
  | { kind: "done"; done: Done }
  | { kind: "error"; message: string } // the server sent an `error` event
  | { kind: "rejected"; reason: "empty" | "loading" | "load_error" | "busy" | "health_failed"; message: string }
  | { kind: "disconnected"; message: string } // stream failed to open or dropped before `done`
  | { kind: "invalid"; message: string } // an event did not match the contract
  | { kind: "aborted" };

/** Called for every contract event, with the raw `data` string exactly as it arrived. */
export type AskListener = (event: AskEvent, raw: string) => void;

export function createClient(options: ClientOptions = {}) {
  const base = (options.base ?? API_BASE).replace(/\/+$/, "");
  const fetchImpl: typeof fetch = options.fetch ?? ((input, init) => fetch(input, init));

  async function getJson<T>(path: string, parse: (value: unknown) => T, signal?: AbortSignal): Promise<T> {
    const res = await fetchImpl(`${base}${path}`, { signal, headers: { Accept: "application/json" } });
    const body: unknown = await res.json().catch(() => undefined);
    if (!res.ok) {
      const parsed = ErrorBodySchema.safeParse(body);
      throw new ApiError(res.status, parsed.success ? parsed.data.error : `HTTP ${res.status}`);
    }
    return parse(body);
  }

  const getHealth = (signal?: AbortSignal) => getJson("/health", parseHealth, signal);
  const getQuestions = (signal?: AbortSignal) => getJson("/questions", parseQuestions, signal);
  const getDocument = (docId: string, signal?: AbortSignal) =>
    getJson(`/documents/${encodeURIComponent(docId)}`, parseDocument, signal);

  /**
   * Runs one question over SSE. Checks /health first (EventSource cannot read 409/503 bodies), then opens an
   * EventSource and closes it on `done`, `error`, a dropped connection, a contract violation or `signal` abort, so
   * EventSource never auto-reconnects and silently re-runs the question.
   */
  async function ask(question: string, onEvent: AskListener, signal?: AbortSignal): Promise<AskOutcome> {
    if (!question.trim()) return { kind: "rejected", reason: "empty", message: "missing q" };
    if (signal?.aborted) return { kind: "aborted" };

    let health: Health;
    try {
      health = await getHealth(signal);
    } catch (e) {
      if (signal?.aborted) return { kind: "aborted" };
      return { kind: "rejected", reason: "health_failed", message: `health check failed: ${errorMessage(e)}` };
    }
    if (health.status === "loading") {
      return { kind: "rejected", reason: "loading", message: "indexes are still loading" };
    }
    if (health.status === "error") {
      return { kind: "rejected", reason: "load_error", message: health.error ?? "the backend failed to load" };
    }
    if (health.busy) return { kind: "rejected", reason: "busy", message: "another question is running" };
    if (signal?.aborted) return { kind: "aborted" };

    const EventSourceImpl = options.EventSource ?? globalThis.EventSource;
    const source = new EventSourceImpl(`${base}/ask?q=${encodeURIComponent(question)}`);

    return new Promise<AskOutcome>((resolve) => {
      let finished = false;
      let received = 0;
      const finish = (outcome: AskOutcome) => {
        if (finished) return;
        finished = true;
        source.close();
        signal?.removeEventListener("abort", onAbort);
        resolve(outcome);
      };
      const onAbort = () => finish({ kind: "aborted" });
      signal?.addEventListener("abort", onAbort);

      const handle = (name: AskEventName, raw: string) => {
        if (finished) return;
        let event: AskEvent;
        try {
          event = parseAskEvent(name, JSON.parse(raw));
        } catch (e) {
          finish({ kind: "invalid", message: errorMessage(e) });
          return;
        }
        received += 1;
        onEvent(event, raw);
        if (event.event === "done") finish({ kind: "done", done: event.data });
        else if (event.event === "error") finish({ kind: "error", message: event.data.message });
      };

      for (const name of ASK_EVENTS) {
        if (name === "error") continue;
        source.addEventListener(name, (ev) => handle(name, (ev as MessageEvent<string>).data));
      }
      // The server's `error` event and EventSource's own connection error share the name "error". Only the former is
      // a MessageEvent carrying data.
      source.addEventListener("error", (ev) => {
        if (ev instanceof MessageEvent && typeof ev.data === "string") {
          handle("error", ev.data);
          return;
        }
        finish({
          kind: "disconnected",
          message:
            received === 0
              ? "could not open /ask (EventSource hides the status: 400, 409 or 503; check /health)"
              : `connection dropped after ${received} event(s), before done`,
        });
      });
    });
  }

  return { base, getHealth, getQuestions, getDocument, ask };
}

export type ApiClient = ReturnType<typeof createClient>;

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
