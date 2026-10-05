"use client"

import { ChevronDown } from "lucide-react"
import { useId, useState } from "react"

import type { Question } from "@/lib/api"
import { goldDocs } from "@/lib/retrieval"
import type { RunState } from "@/lib/run"
import { cn } from "@/lib/utils"

import { CitationChip } from "./answer-body"

/** The benchmark's gold answer and where each gold document landed, for questions picked from /questions. */
export function GoldAnswer({
  question,
  run,
  active,
  onCite,
}: {
  question: Question
  run: RunState
  active: number | null
  onCite: (n: number) => void
}) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const gold = goldDocs(question, run.events.fuse, run.events.rerank)
  const found = gold.filter((g) => g.status === "top10").length

  return (
    <div className="flex flex-col gap-2" data-role="gold">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1 self-start rounded text-[12px] font-medium text-neutral-600 hover:text-neutral-900"
      >
        {open ? "Hide gold answer" : "Show gold answer"}
        <ChevronDown className={cn("size-4 transition-transform", open && "rotate-180")} aria-hidden />
      </button>
      {open && (
        <div id={id} className="flex flex-col gap-3 rounded-xl bg-neutral-50 px-3.5 py-3 text-[13px] leading-relaxed" data-role="gold-panel">
          <div>
            <p className="text-[10.5px] font-semibold uppercase tracking-[0.09em] text-neutral-600">
              Benchmark gold answer
            </p>
            <p className="mt-1 whitespace-pre-wrap text-neutral-800" data-role="gold-answer">
              {question.gold_answer}
            </p>
          </div>
          <div>
            <p className="text-[10.5px] font-semibold uppercase tracking-[0.09em] text-neutral-600">
              Gold documents{gold.length > 0 && ` · ${found} of ${gold.length} in the top 10`}
            </p>
            {gold.length === 0 ? (
              <p className="mt-1 text-neutral-700">
                None: the benchmark expects this answer is not in the corpus.
              </p>
            ) : (
              <ul className="mt-1 flex flex-col gap-1" data-role="gold-docs">
                {gold.map((g) => (
                  <li key={g.doc_id} className="flex flex-wrap items-center gap-x-2" data-gold={g.doc_id} data-status={g.status}>
                    {g.status === "top10" ? (
                      <>
                        <span className="rounded-full bg-neutral-800 px-2 py-0.5 text-[10.5px] font-bold uppercase tracking-[0.05em] text-white">
                          Top 10
                        </span>
                        <span className="text-neutral-700">rank</span>
                        <CitationChip n={g.rank} active={active === g.rank} onCite={onCite} label={g.title} />
                        <span className="min-w-0 truncate font-medium text-neutral-800">{g.title}</span>
                      </>
                    ) : (
                      <>
                        <span className="rounded-full bg-neutral-200 px-2 py-0.5 text-[10.5px] font-bold uppercase tracking-[0.05em] text-neutral-800">
                          Missed
                        </span>
                        <span className="text-neutral-700">
                          {g.fusedRank !== null
                            ? `fused #${g.fusedRank} of 100, not kept by the reranker`
                            : "not among the 100 retrieved candidates"}
                        </span>
                        {g.title && <span className="min-w-0 truncate font-medium text-neutral-800">{g.title}</span>}
                        <span className="font-mono text-[11px] text-neutral-600">{g.doc_id}</span>
                      </>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
