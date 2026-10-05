"use client";

import { useEffect, useRef, useState } from "react";

import { createClient, isMockMode, type AskOutcome, type Question } from "@/lib/api";

const api = createClient();

/** Runs one question and prints each raw event, stamped with seconds since the request started. */
async function runQuestion(question: string, print: (line: string) => void, signal: AbortSignal): Promise<void> {
  const started = performance.now();
  const stamp = () => `[+${((performance.now() - started) / 1000).toFixed(3)}s]`;
  print(`${stamp()} GET ${api.base}/ask?q=${encodeURIComponent(question)}`);
  const outcome: AskOutcome = await api.ask(
    question,
    (event, raw) => print(`${stamp()} event: ${event.event}\ndata: ${raw}`),
    signal,
  );
  print(`${stamp()} outcome: ${JSON.stringify(outcome)}`);
}

export default function Home() {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    api.getQuestions(controller.signal).then(setQuestions, (e: unknown) => {
      if (!controller.signal.aborted) setLoadError(e instanceof Error ? e.message : String(e));
    });
    return () => {
      controller.abort();
      abort.current?.abort();
    };
  }, []);

  async function run(q: Question) {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setRunning(true);
    setLog([]);
    await runQuestion(q.question, (line) => setLog((lines) => [...lines, line]), controller.signal);
    if (abort.current === controller) setRunning(false);
  }

  return (
    <main className="mx-auto max-w-5xl p-4 font-mono text-sm">
      <h1 className="mb-2 text-base font-bold">Enterprise search demo</h1>
      <p className="mb-4">
        API base: {api.base} ({isMockMode(api.base) ? "mock mode, recorded fixtures" : "live mode"})
      </p>
      {loadError && <p className="mb-4">Could not load questions: {loadError}</p>}
      <ul className="mb-4 space-y-1">
        {questions.map((q) => (
          <li key={q.question_id}>
            <button
              type="button"
              className="text-left underline disabled:opacity-50"
              disabled={running}
              onClick={() => void run(q)}
            >
              {q.question_id} [{q.question_type}] {q.question}
            </button>
          </li>
        ))}
      </ul>
      {running && (
        <button type="button" className="mb-4 underline" onClick={() => abort.current?.abort()}>
          stop
        </button>
      )}
      <pre className="whitespace-pre-wrap break-all">{log.join("\n\n")}</pre>
    </main>
  );
}
