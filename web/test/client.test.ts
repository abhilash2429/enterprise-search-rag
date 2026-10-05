// Paths no fixture covers, built from docs/demo-api.md: a refusal run, a run without router and verifier, server
// errors, refused or dropped connections, contract drift and aborts.

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { createClient, type AskEvent, type Generate, type Health, type Verify } from "@/lib/api";
import { FIXTURES_DIR, sseFrame, type RecordedEvent } from "@/lib/mock";

import { fakeEventSource, stubFetch } from "./harness";

const READY: Health = JSON.parse(readFileSync(path.join(FIXTURES_DIR, "health.json"), "utf8"));

// The info_not_found run supplies realistic start..rerank payloads; generate, verify and done are written from the
// contract for a refusal.
const INFO_NOT_FOUND: RecordedEvent[] = readFileSync(path.join(FIXTURES_DIR, "ask", "qst_0484.jsonl"), "utf8")
  .split(/\r?\n/)
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l) as RecordedEvent);
const prefix = (until: string) => INFO_NOT_FOUND.slice(0, INFO_NOT_FOUND.findIndex((e) => e.event === until));
const question = (INFO_NOT_FOUND[0].data as { question: string }).question;

function run(health: unknown, frames: [string, unknown][] | null) {
  const fetchImpl = stubFetch(health, frames && frames.map(([e, d]) => sseFrame(e, d)).join(""));
  const es = fakeEventSource(fetchImpl);
  const client = createClient({ base: "/mock-api", fetch: fetchImpl, EventSource: es.EventSource });
  const events: AskEvent[] = [];
  return { client, events, opened: es.opened, listener: (e: AskEvent) => events.push(e) };
}

describe("synthetic refusal run", () => {
  const generated = (INFO_NOT_FOUND.find((e) => e.event === "generate")!.data as Generate).context;
  const generate: Generate = {
    answer: "The provided documents do not contain the information needed to answer this question.",
    abstained: true,
    partial: false,
    citations: [],
    context: generated,
    version_pairs: [],
    seconds: 3.1,
    cost_usd: 0.0029,
  };
  const verify: Verify = { confidence: null, skipped: "abstained", seconds: 0, cost_usd: 0 };
  const done = {
    seconds: { route: 1.8, retrieve: 4.3, fetch: 0.1, rerank: 33.8, generate: 3.1, total: 43.1 },
    cost_usd: 0.0029,
  };
  const frames: [string, unknown][] = [
    ...prefix("generate").map((e): [string, unknown] => [e.event, e.data]),
    ["generate", generate],
    ["verify", verify],
    ["done", done],
  ];

  it("parses every event, including generate.abstained and verify.skipped", async () => {
    const { client, events, listener } = run(READY, frames);
    const outcome = await client.ask(question, listener);
    expect(outcome).toEqual({ kind: "done", done });
    expect(events.map((e) => e.event)).toEqual(["start", "route", "retrieve", "fuse", "rerank", "generate", "verify", "done"]);
    const g = events.find((e) => e.event === "generate");
    const v = events.find((e) => e.event === "verify");
    expect(g?.data).toEqual(generate);
    expect(v?.data).toEqual({ confidence: null, skipped: "abstained", seconds: 0, cost_usd: 0 });
  });
});

describe("synthetic run without router and verifier", () => {
  it("parses null routed lists and stage times without route and verify", async () => {
    const config = { router: false, rerank: true, dense: "binary", verify: false };
    const ids = ["dsid_a", "dsid_b"];
    const { client, events, listener } = run(READY, [
      ["start", { question: "q", config, stages: ["retrieve", "fuse", "rerank", "generate"] }],
      ["retrieve", { lists: { bm25: ids, dense: ids, bm25_routed: null, dense_routed: null }, seconds: 3.5 }],
      [
        "fuse",
        {
          candidates: ids.map((doc_id, i) => ({
            doc_id,
            source: "slack",
            title: doc_id,
            rrf_score: 2 / (61 + i),
            ranks: { bm25: i + 1, dense: i + 1, bm25_routed: null, dense_routed: null },
          })),
          seconds: 0.1,
        },
      ],
      [
        "rerank",
        {
          hits: ids.map((doc_id, i) => ({
            rank: i + 1,
            doc_id,
            source: "slack",
            title: doc_id,
            snippet: "",
            rerank_score: 0.5,
            fused_rank: i + 1,
          })),
          order: ids.map((doc_id) => ({ doc_id, rerank_score: 0.5 })),
          seconds: 9,
        },
      ],
      [
        "generate",
        {
          answer: "Yes [1].",
          abstained: false,
          partial: false,
          citations: [{ n: 1, doc_id: "dsid_a", source: "slack", title: "dsid_a" }],
          context: ids.map((doc_id, i) => ({ n: i + 1, doc_id, source: "slack", title: doc_id })),
          version_pairs: [{ a: 1, b: 2, cosine: 0.93 }],
          seconds: 2,
          cost_usd: 0.002,
        },
      ],
      ["done", { seconds: { retrieve: 3.5, fetch: 0.1, rerank: 9, generate: 2, total: 14.6 }, cost_usd: 0.002 }],
    ]);
    expect((await client.ask("q", listener)).kind).toBe("done");
    expect(events.map((e) => e.event)).toEqual(["start", "retrieve", "fuse", "rerank", "generate", "done"]);
  });

  it("parses an unparsed verifier reply (flagged null, error set)", async () => {
    const { client, events, listener } = run(READY, [
      ["verify", { confidence: { flagged: null, rerank_top: null, error: "no JSON" }, skipped: null, seconds: 5, cost_usd: 0.001 }],
      ["done", { seconds: { retrieve: 1, fetch: 0, generate: 1, verify: 5, total: 7 }, cost_usd: 0.001 }],
    ]);
    expect((await client.ask("q", listener)).kind).toBe("done");
    expect(events[0]).toMatchObject({ event: "verify", data: { confidence: { flagged: null, error: "no JSON" } } });
  });
});

describe("failures", () => {
  it("ends on the server's error event and ignores anything after it", async () => {
    const { client, events, listener, opened } = run(READY, [
      ...prefix("fuse").map((e): [string, unknown] => [e.event, e.data]),
      ["error", { message: "RuntimeError: CUDA out of memory" }],
      ["done", { seconds: { retrieve: 1, fetch: 0, generate: 1, total: 2 }, cost_usd: 0 }],
    ]);
    const outcome = await client.ask(question, listener);
    expect(outcome).toEqual({ kind: "error", message: "RuntimeError: CUDA out of memory" });
    expect(events.map((e) => e.event)).toEqual(["start", "route", "retrieve", "error"]);
    expect(opened[0].closed).toBe(true);
  });

  it("reports a stream that ends before done as disconnected, and closes it so it cannot reconnect", async () => {
    const { client, events, listener, opened } = run(
      READY,
      prefix("rerank").map((e): [string, unknown] => [e.event, e.data]),
    );
    const outcome = await client.ask(question, listener);
    expect(outcome.kind).toBe("disconnected");
    expect(events).toHaveLength(4);
    expect(opened).toHaveLength(1);
    expect(opened[0].closed).toBe(true);
  });

  it("reports a refused stream (409 race after the health check) as disconnected", async () => {
    const { client, listener } = run(READY, null);
    const outcome = await client.ask(question, listener);
    expect(outcome).toMatchObject({ kind: "disconnected", message: expect.stringContaining("409") });
  });

  it.each([
    [{ ...READY, busy: true }, "busy"],
    [{ ...READY, status: "loading" }, "loading"],
    [{ ...READY, status: "error", error: "FileNotFoundError: data/index" }, "load_error"],
  ] as const)("checks /health before opening the stream (%o)", async (health, reason) => {
    const { client, listener, opened } = run(health, []);
    expect(await client.ask(question, listener)).toMatchObject({ kind: "rejected", reason });
    expect(opened).toHaveLength(0);
  });

  it("rejects an empty question without a request", async () => {
    const { client, listener, opened } = run(READY, []);
    expect(await client.ask("   ", listener)).toMatchObject({ kind: "rejected", reason: "empty" });
    expect(opened).toHaveLength(0);
  });

  it("rejects a health body that does not match the contract", async () => {
    const { client, listener } = run({ ...READY, extra: 1 }, []);
    expect(await client.ask(question, listener)).toMatchObject({ kind: "rejected", reason: "health_failed" });
  });

  it.each([
    ["an unknown key", ["route", { sources: ["slack"], seconds: 1, extra: true }]],
    ["an unknown source", ["route", { sources: ["notion"], seconds: 1 }]],
    ["an unknown claim verdict", ["verify", { confidence: { flagged: true, rerank_top: 0.5, claims: [{ claim: "c", cited: [1], verdict: "maybe" }] }, skipped: null, seconds: 1, cost_usd: 0 }]],
    ["a missing field", ["done", { seconds: { retrieve: 1, fetch: 0, generate: 1 }, cost_usd: 0 }]],
  ] as const)("reports contract drift (%s) as invalid", async (_what, frame) => {
    const { client, listener, opened } = run(READY, [frame as [string, unknown]]);
    const outcome = await client.ask(question, listener);
    expect(outcome.kind).toBe("invalid");
    expect(opened[0].closed).toBe(true);
  });

  it("closes the stream on abort", async () => {
    const { client, listener, opened } = run(READY, prefix("rerank").map((e): [string, unknown] => [e.event, e.data]));
    const controller = new AbortController();
    const pending = client.ask(question, (e) => {
      listener(e);
      controller.abort();
    }, controller.signal);
    expect(await pending).toEqual({ kind: "aborted" });
    expect(opened[0].closed).toBe(true);
  });
});
