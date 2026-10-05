// Every recorded fixture replays through the client against the mock-api route handlers, and every payload parses
// into its contract type.

import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { ASK_EVENTS, parseDocument, parseHealth, parseQuestions, type AskEvent, type Question } from "@/lib/api";
import { FIXTURES_DIR, UNKNOWN_QUESTION_MESSAGE, type RecordedEvent } from "@/lib/mock";

import { mockClient, mockFetch } from "./harness";

const read = (...parts: string[]) => readFileSync(path.join(FIXTURES_DIR, ...parts), "utf8");
const questions: Question[] = parseQuestions(JSON.parse(read("questions.json")));
const recorded = (id: string): RecordedEvent[] =>
  read("ask", `${id}.jsonl`)
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as RecordedEvent);

let previousSpeed: string | undefined;
beforeEach(() => {
  previousSpeed = process.env.NEXT_PUBLIC_MOCK_SPEED;
  process.env.NEXT_PUBLIC_MOCK_SPEED = "2000";
});
afterEach(() => {
  if (previousSpeed === undefined) delete process.env.NEXT_PUBLIC_MOCK_SPEED;
  else process.env.NEXT_PUBLIC_MOCK_SPEED = previousSpeed;
});

describe("static fixtures", () => {
  it("health.json is a ready Health", () => {
    expect(parseHealth(JSON.parse(read("health.json")))).toMatchObject({ status: "ready", busy: false });
  });

  it("questions.json is a Question[] with a recorded run for each", () => {
    expect(questions.length).toBeGreaterThan(0);
    const runs = readdirSync(path.join(FIXTURES_DIR, "ask")).sort();
    expect(runs).toEqual(questions.map((q) => `${q.question_id}.jsonl`).sort());
  });

  it("every documents/<id>.json is a Document with a matching id", () => {
    const files = readdirSync(path.join(FIXTURES_DIR, "documents"));
    expect(files.length).toBeGreaterThan(0);
    for (const file of files) {
      expect(parseDocument(JSON.parse(read("documents", file))).doc_id).toBe(path.basename(file, ".json"));
    }
  });
});

describe.each(questions.map((q) => [q.question_id, q] as const))("recorded run %s", (id, q) => {
  it("replays through the client, in contract order, each event parsing into its type", async () => {
    const { client, opened } = mockClient();
    const events: AskEvent[] = [];
    const outcome = await client.ask(q.question, (e) => events.push(e));

    expect(outcome.kind).toBe("done");
    expect(opened).toHaveLength(1);
    expect(opened[0].closed).toBe(true);

    // Parsed events equal the recorded payloads, one for one.
    const run = recorded(id);
    expect(events.map((e) => [e.event, e.data])).toEqual(run.map((e) => [e.event, e.data]));

    // Order: start, the stages listed in start.stages in contract order, done.
    const start = events[0];
    if (start.event !== "start") throw new Error("first event is not start");
    expect(start.data.question).toBe(q.question);
    const stageOrder = ASK_EVENTS.filter((n) => (start.data.stages as string[]).includes(n));
    expect(events.map((e) => e.event)).toEqual(["start", ...stageOrder, "done"]);
  });

  it("serves a document for every reranked hit", async () => {
    const { client } = mockClient();
    const rerank = recorded(id).find((e) => e.event === "rerank");
    if (!rerank) return;
    const hits = (rerank.data as { hits: { doc_id: string }[] }).hits;
    for (const hit of hits) expect((await client.getDocument(hit.doc_id)).doc_id).toBe(hit.doc_id);
  });
});

describe("mock /ask", () => {
  it("paces events at their recorded t divided by NEXT_PUBLIC_MOCK_SPEED", async () => {
    process.env.NEXT_PUBLIC_MOCK_SPEED = "100";
    const q = questions.reduce((a, b) => (recorded(a.question_id).at(-1)!.t < recorded(b.question_id).at(-1)!.t ? a : b));
    const run = recorded(q.question_id);
    const { client } = mockClient();
    const t0 = performance.now();
    const arrivals: number[] = [];
    expect((await client.ask(q.question, () => arrivals.push(performance.now()))).kind).toBe("done");
    expect(arrivals).toHaveLength(run.length);
    const first = arrivals[0];
    expect(first - t0).toBeLessThan(150);
    run.forEach((e, i) => {
      const expected = (e.t * 1000) / 100;
      expect(arrivals[i] - first).toBeGreaterThanOrEqual(expected - 25);
      expect(arrivals[i] - first).toBeLessThan(expected + 250);
    });
  });

  it("streams one error event for a question with no recorded run", async () => {
    const { client } = mockClient();
    const events: AskEvent[] = [];
    const outcome = await client.ask("What is the meaning of life?", (e) => events.push(e));
    expect(events).toEqual([{ event: "error", data: { message: UNKNOWN_QUESTION_MESSAGE } }]);
    expect(outcome).toEqual({ kind: "error", message: UNKNOWN_QUESTION_MESSAGE });
  });

  it("matches the question text after trimming, like the backend", async () => {
    const { client } = mockClient();
    expect((await client.ask(`  ${questions[0].question}\n`, () => {})).kind).toBe("done");
  });

  it("answers 400 with an error body when q is missing", async () => {
    const res = await mockFetch("/mock-api/ask?q=");
    expect(res.status).toBe(400);
    expect(await res.json()).toEqual({ error: "missing q" });
  });

  it("answers 404 with an error body for a document with no fixture", async () => {
    const { client } = mockClient();
    await expect(client.getDocument("dsid_missing")).rejects.toMatchObject({ status: 404 });
    await expect(client.getDocument("../questions")).rejects.toMatchObject({ status: 404 });
  });
});
