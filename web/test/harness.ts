// Test harness: runs the client against the mock-api route handlers in process, with no Next server and no browser.

import { GET as askRoute } from "@/app/mock-api/ask/route";
import { GET as documentRoute } from "@/app/mock-api/documents/[doc_id]/route";
import { GET as healthRoute } from "@/app/mock-api/health/route";
import { GET as questionsRoute } from "@/app/mock-api/questions/route";
import { createClient, type EventSourceConstructor, type EventSourceLike } from "@/lib/api";

const ORIGIN = "http://mock.test";

/** A fetch that serves /mock-api/* from the route handlers. */
export const mockFetch: typeof fetch = async (input, init) => {
  const url = new URL(String(input), ORIGIN);
  const request = new Request(url, { signal: init?.signal ?? undefined });
  const p = url.pathname;
  if (p === "/mock-api/health") return healthRoute();
  if (p === "/mock-api/questions") return questionsRoute();
  if (p === "/mock-api/ask") return askRoute(request);
  const doc = /^\/mock-api\/documents\/([^/]+)$/.exec(p);
  if (doc) return documentRoute(request, { params: Promise.resolve({ doc_id: decodeURIComponent(doc[1]) }) });
  return new Response("not found", { status: 404 });
};

/**
 * Minimal browser-like EventSource over a fetch: a non-200 or non-SSE response, and the end of the stream, dispatch a
 * plain `error` Event (a browser would then reconnect unless closed); each SSE message dispatches a MessageEvent.
 */
export function fakeEventSource(fetchImpl: typeof fetch) {
  const opened: FakeEventSource[] = [];
  class FakeEventSource extends EventTarget implements EventSourceLike {
    closed = false;
    private readonly controller = new AbortController();
    constructor(readonly url: string) {
      super();
      opened.push(this);
      void this.pump();
    }
    close() {
      this.closed = true;
      this.controller.abort();
    }
    private emit(event: Event) {
      if (!this.closed) this.dispatchEvent(event);
    }
    private async pump() {
      try {
        const res = await fetchImpl(this.url, { signal: this.controller.signal });
        if (!res.ok || !res.headers.get("content-type")?.startsWith("text/event-stream") || !res.body) {
          this.emit(new Event("error"));
          return;
        }
        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += value;
          let end: number;
          while ((end = buffer.indexOf("\n\n")) !== -1) {
            this.dispatchFrame(buffer.slice(0, end));
            buffer = buffer.slice(end + 2);
          }
        }
        this.emit(new Event("error"));
      } catch {
        this.emit(new Event("error"));
      }
    }
    private dispatchFrame(frame: string) {
      let name = "message";
      const data: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith(":")) continue;
        const i = line.indexOf(":");
        const field = i === -1 ? line : line.slice(0, i);
        const value = i === -1 ? "" : line.slice(i + 1).replace(/^ /, "");
        if (field === "event") name = value;
        else if (field === "data") data.push(value);
      }
      if (data.length) this.emit(new MessageEvent(name, { data: data.join("\n") }));
    }
  }
  return { EventSource: FakeEventSource as EventSourceConstructor, opened };
}

export function mockClient(fetchImpl: typeof fetch = mockFetch) {
  const es = fakeEventSource(fetchImpl);
  return { client: createClient({ base: "/mock-api", fetch: fetchImpl, EventSource: es.EventSource }), opened: es.opened };
}

/** A fetch whose /health returns `health` and whose /ask streams `body` (raw SSE text). */
export function stubFetch(health: unknown, body: string | null): typeof fetch {
  return async (input) => {
    const p = new URL(String(input), ORIGIN).pathname;
    if (p === "/mock-api/health") return Response.json(health);
    if (p === "/mock-api/ask" && body !== null) {
      return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
    }
    return Response.json({ error: "another question is running" }, { status: 409 });
  };
}
