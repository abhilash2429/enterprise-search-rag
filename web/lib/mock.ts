// Mock mode: serves the recorded runs in web/fixtures/ (docs/demo-api.md, "Fixtures") from Next route handlers.
// Server-only: reads the fixtures from disk.

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";

export const FIXTURES_DIR = path.join(process.cwd(), "fixtures");

/** One line of ask/<question_id>.jsonl: `t` is seconds since the request started. */
export type RecordedEvent = { t: number; event: string; data: unknown };

export const UNKNOWN_QUESTION_MESSAGE =
  "Only the demo questions work offline: mock mode replays recorded runs and has none for this question.";

const DOC_ID = /^[A-Za-z0-9_-]+$/;

export function fixturePath(...parts: string[]): string {
  return path.join(FIXTURES_DIR, ...parts);
}

export async function readFixtureText(...parts: string[]): Promise<string> {
  return readFile(fixturePath(...parts), "utf8");
}

export function jsonFileResponse(text: string, status = 200): Response {
  return new Response(text, { status, headers: { "Content-Type": "application/json" } });
}

export function errorResponse(status: number, error: string): Response {
  return Response.json({ error }, { status });
}

export function documentFile(docId: string): string | null {
  if (!DOC_ID.test(docId)) return null;
  const file = fixturePath("documents", `${docId}.json`);
  return existsSync(file) ? file : null;
}

/** Finds the recorded run whose question text equals `q` (both trimmed, as the backend strips `q`). */
export async function findRecordedQuestionId(q: string): Promise<string | null> {
  const questions = JSON.parse(await readFixtureText("questions.json")) as { question_id: string; question: string }[];
  const wanted = q.trim();
  return questions.find((x) => x.question.trim() === wanted)?.question_id ?? null;
}

export async function readRecordedRun(questionId: string): Promise<RecordedEvent[]> {
  const text = await readFixtureText("ask", `${questionId}.jsonl`);
  return text
    .split(/\r?\n/)
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as RecordedEvent);
}

/** NEXT_PUBLIC_MOCK_SPEED: replay speed-up factor, default 1 (real pacing). */
export function mockSpeed(value: string | undefined = process.env.NEXT_PUBLIC_MOCK_SPEED): number {
  const speed = Number(value ?? "1");
  return Number.isFinite(speed) && speed > 0 ? speed : 1;
}

export function sseFrame(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

/** Streams `events` as SSE, each at `t / speed` seconds after the stream starts, then ends the stream. */
export function replay(events: RecordedEvent[], speed: number, signal?: AbortSignal): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const timers: ReturnType<typeof setTimeout>[] = [];
  const stop = () => timers.forEach(clearTimeout);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      if (events.length === 0) {
        controller.close();
        return;
      }
      signal?.addEventListener("abort", stop);
      let sent = 0;
      for (const e of events) {
        timers.push(
          setTimeout(
            () => {
              controller.enqueue(encoder.encode(sseFrame(e.event, e.data)));
              sent += 1;
              if (sent === events.length) {
                signal?.removeEventListener("abort", stop);
                controller.close();
              }
            },
            (e.t * 1000) / speed,
          ),
        );
      }
    },
    cancel: stop,
  });
}

export function sseResponse(body: ReadableStream<Uint8Array>): Response {
  return new Response(body, {
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
