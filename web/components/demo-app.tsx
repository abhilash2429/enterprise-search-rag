"use client"

import { ArrowUp, Square } from "lucide-react"
import { useCallback, useEffect, useReducer, useRef, useState } from "react"

import { DocumentViewer, type ViewerState } from "@/components/document-viewer"
import { EvidenceRail, type Hit } from "@/components/evidence-rail"
import { PipelinePanel } from "@/components/pipeline-panel"
import { Sidebar } from "@/components/sidebar"
import { TurnView } from "@/components/turn-view"
import { Button } from "@/components/ui/button"
import { ChatContainerContent, ChatContainerRoot } from "@/components/ui/chat-container"
import { PromptInput, PromptInputActions, PromptInputTextarea } from "@/components/ui/prompt-input"
import { PromptSuggestion } from "@/components/ui/prompt-suggestion"
import { createClient, isMockMode, type Health, type Question } from "@/lib/api"
import { IDLE, runReducer, type RunAction, type RunState } from "@/lib/run"

const api = createClient()
const MOCK = isMockMode(api.base)

let nextRunId = 1

/** Starts one run; events and the outcome are stamped with the client clock as they arrive. */
function startRun(question: string, dispatch: (a: RunAction) => void, signal: AbortSignal): void {
  const id = nextRunId++
  dispatch({ type: "begin", id, question, at: performance.now() })
  void api
    .ask(question, (event) => dispatch({ type: "event", id, event, at: performance.now() }), signal)
    .then((outcome) => dispatch({ type: "finish", id, outcome, at: performance.now() }))
}

export function DemoApp() {
  const [run, dispatch] = useReducer(runReducer, IDLE)
  const [history, setHistory] = useState<RunState[]>([])
  const [input, setInput] = useState("")
  const [questions, setQuestions] = useState<Question[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [apiDown, setApiDown] = useState(false)
  const [evidenceTurnId, setEvidenceTurnId] = useState<number | null>(null)
  const [highlight, setHighlight] = useState<{ turnId: number; n: number; nonce: number } | null>(null)
  const [viewer, setViewer] = useState<ViewerState | null>(null)
  const abort = useRef<AbortController | null>(null)
  const viewerRequest = useRef(0)

  useEffect(() => {
    const controller = new AbortController()
    api.getQuestions(controller.signal).then(setQuestions, () => {
      if (!controller.signal.aborted) setApiDown(true)
    })
    api.getHealth(controller.signal).then(setHealth, () => {
      if (!controller.signal.aborted) setApiDown(true)
    })
    return () => {
      controller.abort()
      abort.current?.abort()
    }
  }, [])

  const running = run.phase === "checking" || run.phase === "streaming"
  const thread = run.phase === "idle" ? history : [...history, run]
  const latest = thread.at(-1) ?? null
  const evidenceTurn = thread.find((t) => t.id === evidenceTurnId) ?? latest
  const asked = new Set(thread.map((t) => t.question))

  const submit = (text: string) => {
    const question = text.trim()
    if (!question || running) return
    if (run.phase !== "idle") setHistory((h) => [...h, run])
    setInput("")
    setEvidenceTurnId(null)
    setHighlight(null)
    const controller = new AbortController()
    abort.current = controller
    startRun(question, dispatch, controller.signal)
  }

  const newSession = () => {
    abort.current?.abort()
    setHistory([])
    setEvidenceTurnId(null)
    setHighlight(null)
    dispatch({ type: "reset" })
  }

  const cite = useCallback((turnId: number, n: number) => {
    setEvidenceTurnId(turnId)
    setHighlight((h) => ({ turnId, n, nonce: (h?.nonce ?? 0) + 1 }))
  }, [])

  const openDocument = (hit: Hit) => {
    const request = ++viewerRequest.current
    const base = { rank: hit.rank, score: hit.rerank_score, doc_id: hit.doc_id, title: hit.title, source: hit.source }
    setViewer({ ...base, doc: null, error: null })
    api.getDocument(hit.doc_id).then(
      (doc) => request === viewerRequest.current && setViewer({ ...base, doc, error: null }),
      (e: unknown) =>
        request === viewerRequest.current &&
        setViewer({ ...base, doc: null, error: e instanceof Error ? e.message : String(e) })
    )
  }
  const closeViewer = useCallback(() => {
    viewerRequest.current += 1
    setViewer(null)
  }, [])

  const railHighlight = highlight && evidenceTurn && highlight.turnId === evidenceTurn.id ? highlight : null

  return (
    <div className="app-canvas min-h-dvh p-3 text-neutral-700 lg:p-4">
      <div className="mx-auto grid min-h-[calc(100dvh-1.5rem)] max-w-[1680px] grid-cols-1 gap-3 lg:grid-cols-[240px_minmax(480px,1fr)] xl:h-[calc(100dvh-2rem)] xl:min-h-0 xl:grid-cols-[240px_minmax(480px,1fr)_400px]">
        <Sidebar
          questions={questions}
          asked={asked}
          running={running}
          canReset={thread.length > 0}
          onAsk={submit}
          onNewSession={newSession}
        />

        <main className="flex min-h-[720px] min-w-0 flex-col overflow-hidden rounded-[24px] bg-white xl:h-full xl:min-h-0">
          {apiDown && (
            <div className="px-6 pt-5">
              <p className="rounded-lg bg-neutral-200 px-3 py-2 text-[13px] font-medium text-neutral-800">
                The search service is unreachable at {api.base}. Start the backend, or use mock mode.
              </p>
            </div>
          )}
          <header className="px-7 pb-5 pt-5">
            <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-neutral-400">
              EnterpriseRAG-Bench corpus
            </p>
            <div className="mt-1 flex items-end justify-between gap-4">
              <div>
                <h1 className="text-[24px] font-semibold tracking-[-0.03em] text-neutral-800">Company knowledge</h1>
                <p className="mt-1 text-[12px] text-muted-foreground">
                  511,958 documents · Slack, Gmail, Drive, Confluence, Jira, Linear, GitHub, HubSpot, Fireflies
                </p>
              </div>
              <span
                className="rounded-full bg-neutral-100 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.09em] text-neutral-500"
                data-role="mode"
              >
                {MOCK ? "Recorded replay" : health ? `Live · ${health.status}` : "Live"}
              </span>
            </div>
          </header>

          <ChatContainerRoot className="min-h-0 flex-1 px-7" data-role="answer-column">
            <ChatContainerContent className="mx-auto flex max-w-[820px] flex-col gap-5 py-6">
              {thread.length === 0 && (
                <div className="flex flex-col gap-2 py-6">
                  <p className="mb-1 text-[10px] font-semibold uppercase tracking-[0.15em] text-neutral-400">
                    Ask across 9 sources
                  </p>
                  <h2 className="text-[34px] font-semibold leading-tight tracking-[-0.04em] text-neutral-900">
                    Ask your company&apos;s documents.
                  </h2>
                  <p className="mb-4 max-w-[640px] text-[14px] leading-relaxed text-muted-foreground">
                    Each question is routed to likely sources, searched with BM25 and dense retrieval, reranked to the
                    10 best documents, answered with citations, and checked claim by claim. Open any citation to read
                    the source.
                  </p>
                  {questions.slice(0, 4).map((q) => (
                    <PromptSuggestion key={q.question_id} onClick={() => submit(q.question)}>
                      {q.question}
                    </PromptSuggestion>
                  ))}
                </div>
              )}
              {thread.map((t) => (
                <TurnView
                  key={t.id}
                  run={t}
                  active={highlight?.turnId === t.id ? highlight.n : null}
                  onCite={cite}
                  onShowSources={setEvidenceTurnId}
                />
              ))}
            </ChatContainerContent>
          </ChatContainerRoot>

          <div className="bg-white px-7 pb-5 pt-3">
            <PromptInput
              value={input}
              onValueChange={setInput}
              isLoading={running}
              onSubmit={() => submit(input)}
              className="mx-auto max-w-[820px] border-0 bg-neutral-100 shadow-none"
            >
              <PromptInputTextarea aria-label="Question" placeholder="Ask a question about the company's documents" />
              <PromptInputActions className="justify-between px-1 pt-2">
                <span className="text-[10.5px] font-medium text-muted-foreground">
                  {MOCK ? "Recorded replay: the benchmark questions work offline" : "Searching all 9 sources"}
                </span>
                <Button
                  size="icon"
                  aria-label={running ? "Stop" : "Send"}
                  disabled={!running && !input.trim()}
                  className="size-9 rounded-full bg-neutral-800 hover:bg-neutral-700"
                  onClick={() => (running ? abort.current?.abort() : submit(input))}
                >
                  {running ? <Square className="size-4 fill-white text-white" /> : <ArrowUp className="size-5 text-white" />}
                </Button>
              </PromptInputActions>
            </PromptInput>
          </div>
        </main>

        <aside className="flex min-h-[480px] min-w-0 flex-col overflow-hidden rounded-[24px] bg-neutral-100 lg:col-span-2 lg:min-h-0 xl:col-span-1 xl:h-full">
          <div key={evidenceTurn?.id ?? 0} className="min-h-0 flex-1 overflow-y-auto px-4 py-5" data-role="evidence-column">
            <EvidenceRail run={evidenceTurn} highlight={railHighlight} onOpen={openDocument} />
          </div>
          <div className="p-4 pt-2">
            <PipelinePanel health={health} base={api.base} mock={MOCK} />
          </div>
        </aside>
      </div>
      <DocumentViewer state={viewer} onClose={closeViewer} />
    </div>
  )
}
