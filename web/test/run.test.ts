import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { parseAskEvent } from "@/lib/api";
import { FIXTURES_DIR, type RecordedEvent } from "@/lib/mock";
import { IDLE, runReducer, stageViews, type RunState } from "@/lib/run";

const events = readFileSync(path.join(FIXTURES_DIR, "ask", "qst_0147.jsonl"), "utf8")
  .split(/\r?\n/)
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l) as RecordedEvent);

const statuses = (s: RunState) => stageViews(s).map((v) => `${v.stage}:${v.status}`);

describe("stage views", () => {
  it("shows the six stages pending before start", () => {
    const s = runReducer(IDLE, { type: "begin", id: 1, question: "q", at: 0 });
    expect(statuses(s)).toEqual(["route", "retrieve", "fuse", "rerank", "generate", "verify"].map((x) => `${x}:pending`));
  });

  it("marks the stage after the last completed event running, with real seconds on done stages", () => {
    let s = runReducer(IDLE, { type: "begin", id: 1, question: "q", at: 0 });
    const seen: string[][] = [];
    for (const e of events) {
      s = runReducer(s, { type: "event", id: 1, event: parseAskEvent(e.event, e.data), at: e.t * 1000 });
      seen.push(statuses(s));
    }
    expect(seen[0]).toEqual(["route:running", "retrieve:pending", "fuse:pending", "rerank:pending", "generate:pending", "verify:pending"]);
    expect(seen[3]).toEqual(["route:done", "retrieve:done", "fuse:done", "rerank:running", "generate:pending", "verify:pending"]);
    // The running stage's timer starts when the previous event arrived.
    const fuse = events.find((e) => e.event === "fuse")!;
    const views = stageViews(runReducer(IDLE, { type: "begin", id: 1, question: "q", at: 0 }));
    expect(views[0].since).toBeNull();
    let partial = runReducer(IDLE, { type: "begin", id: 1, question: "q", at: 0 });
    for (const e of events.slice(0, 4)) {
      partial = runReducer(partial, { type: "event", id: 1, event: parseAskEvent(e.event, e.data), at: e.t * 1000 });
    }
    expect(stageViews(partial)[3]).toMatchObject({ stage: "rerank", status: "running", since: fuse.t * 1000 });

    s = runReducer(s, { type: "finish", id: 1, outcome: { kind: "done", done: s.events.done! }, at: 0 });
    const done = stageViews(s);
    expect(done.every((v) => v.status === "done")).toBe(true);
    const route = events.find((e) => e.event === "route")!.data as { seconds: number };
    expect(done[0].seconds).toBe(route.seconds);
  });

  it("marks the running stage failed on an error and ignores events from a stale run", () => {
    let s = runReducer(IDLE, { type: "begin", id: 2, question: "q", at: 0 });
    s = runReducer(s, { type: "event", id: 2, event: parseAskEvent("start", events[0].data), at: 0 });
    s = runReducer(s, { type: "event", id: 1, event: parseAskEvent("route", events[1].data), at: 1 });
    expect(s.events.route).toBeUndefined();
    s = runReducer(s, { type: "finish", id: 2, outcome: { kind: "error", message: "boom" }, at: 2 });
    expect(statuses(s)[0]).toBe("route:failed");
    expect(statuses(s)[1]).toBe("retrieve:pending");
  });

  it("marks the running stage stopped on abort", () => {
    let s = runReducer(IDLE, { type: "begin", id: 3, question: "q", at: 0 });
    s = runReducer(s, { type: "event", id: 3, event: parseAskEvent("start", events[0].data), at: 0 });
    s = runReducer(s, { type: "finish", id: 3, outcome: { kind: "aborted" }, at: 1 });
    expect(statuses(s)[0]).toBe("route:stopped");
  });
});
