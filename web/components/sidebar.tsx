"use client"

import { ChartNoAxesColumn, MessageSquareText, Plus } from "lucide-react"
import Link from "next/link"

import { Button } from "@/components/ui/button"
import type { Question } from "@/lib/api"
import { cn } from "@/lib/utils"

export const TYPE_LABEL: Record<string, string> = {
  basic: "Basic",
  semantic: "Semantic",
  conflicting_info: "Conflicting info",
  project_related: "Project",
  info_not_found: "Not in corpus",
  miscellaneous: "Miscellaneous",
  constrained: "Constrained",
  completeness: "Completeness",
  intra_document_reasoning: "Within one document",
}

export function Sidebar({
  questions,
  asked,
  running,
  canReset,
  onAsk,
  onNewSession,
}: {
  questions: Question[]
  asked: Set<string>
  running: boolean
  canReset: boolean
  onAsk: (question: string) => void
  onNewSession: () => void
}) {
  return (
    <aside className="flex min-h-0 flex-col rounded-[24px] bg-neutral-200 px-3 py-4 lg:min-h-[560px] lg:py-5 xl:h-full xl:min-h-0">
      <div className="px-2">
        <p className="text-[20px] font-semibold tracking-[-0.035em] text-neutral-800">Enterprise Search</p>
        <p className="mt-1 text-[10px] font-medium uppercase tracking-[0.14em] text-neutral-600">
          Company knowledge
        </p>
      </div>

      <Link
        href="/benchmark"
        className="mt-4 flex items-center gap-2 rounded-xl px-3 py-2 text-[13px] font-medium text-neutral-700 hover:bg-neutral-300/70"
      >
        <ChartNoAxesColumn className="size-4 shrink-0" aria-hidden />
        Benchmark results
      </Link>

      <nav aria-label="Demo questions" className="mt-4 flex-1 lg:overflow-y-auto">
        <div className="mb-2 px-2">
          <h2 className="text-[10px] font-semibold uppercase tracking-[0.12em] text-neutral-600">Benchmark questions</h2>
        </div>
        <div className="flex flex-col gap-1">
          {questions.map((q, i) => {
            const done = asked.has(q.question)
            return (
              <button
                key={q.question_id}
                type="button"
                data-question={q.question_id}
                aria-label={`${i < 8 ? `Question ${i + 1}, key ${i + 1}: ` : ""}${q.question}`}
                disabled={running}
                onClick={() => onAsk(q.question)}
                className={cn(
                  "w-full rounded-xl px-3 py-2.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-500/40 disabled:cursor-not-allowed",
                  done ? "bg-white text-neutral-900" : "text-neutral-600 hover:bg-neutral-300/70 disabled:hover:bg-transparent"
                )}
              >
                <span className="flex items-start gap-2.5">
                  <MessageSquareText
                    className={cn("mt-0.5 size-4 shrink-0", done ? "text-neutral-700" : "text-neutral-600")}
                    aria-hidden
                  />
                  <span className="min-w-0 flex-1">
                    <span className="line-clamp-2 text-[13px] font-medium leading-snug text-current">
                      {q.question}
                    </span>
                    <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-slate-600">
                      {i < 8 && (
                        <kbd className="dev-only rounded border border-neutral-400 px-1 font-mono text-[10px] leading-tight text-neutral-700">
                          {i + 1}
                        </kbd>
                      )}
                      {TYPE_LABEL[q.question_type] ?? q.question_type}
                      {done && " · asked"}
                    </span>
                  </span>
                </span>
              </button>
            )
          })}
        </div>
      </nav>

      <div className="hidden pt-3 lg:block">
        <Button
          type="button"
          variant="ghost"
          className="h-9 w-full justify-start gap-2 bg-neutral-300/60 text-neutral-700 hover:bg-white"
          disabled={!canReset || running}
          onClick={onNewSession}
        >
          <Plus className="size-4" />
          New session
        </Button>
        <p className="mt-2 px-1 text-[10.5px] leading-snug text-muted-foreground">Clears this conversation.</p>
      </div>
    </aside>
  )
}
