"use client"

import { FileSearch } from "lucide-react"
import { motion, useReducedMotion } from "motion/react"
import { useEffect, useRef } from "react"

import { TextShimmer } from "@/components/ui/text-shimmer"
import type { Rerank } from "@/lib/api"
import type { RunState } from "@/lib/run"
import { cn } from "@/lib/utils"

import { SourceTag } from "./source-dot"
import { progressLabel } from "./stage-trace"

export type Hit = Rerank["hits"][number]

function EvidenceCard({
  hit,
  cited,
  highlighted,
  index,
  onOpen,
}: {
  hit: Hit
  cited: boolean
  highlighted: boolean
  index: number
  onOpen: (hit: Hit) => void
}) {
  const reduceMotion = useReducedMotion() === true
  return (
    <motion.article
      data-rank={hit.rank}
      data-doc={hit.doc_id}
      data-cited={cited}
      data-highlighted={highlighted}
      initial={reduceMotion ? false : { y: 5, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.22, delay: Math.min(index, 5) * 0.045 }}
      className={cn(
        "rounded-[18px] bg-white p-3.5 transition-shadow",
        cited && "bg-neutral-200/70",
        highlighted && "ring-2 ring-neutral-800"
      )}
    >
      <div className="mb-3 flex items-start gap-3">
        <span
          className={cn(
            "flex size-7 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold tabular-nums",
            cited ? "bg-neutral-800 text-white" : "bg-neutral-100 text-neutral-500"
          )}
        >
          {hit.rank}
        </span>
        <div className="min-w-0 flex-1">
          <p className="line-clamp-2 text-[12.5px] font-semibold leading-snug text-neutral-700">{hit.title}</p>
          <p className="mt-1 flex flex-wrap items-center gap-x-1.5 text-[10.5px] font-medium uppercase tracking-[0.08em] text-neutral-400">
            <SourceTag source={hit.source} className="text-neutral-600" />
            <span>· score {hit.rerank_score.toFixed(3)}</span>
            <span>· was #{hit.fused_rank} before reranking</span>
            {cited && <span className="font-bold text-neutral-700">· cited</span>}
          </p>
        </div>
      </div>

      <div className="mb-3 h-px overflow-hidden bg-neutral-100">
        <motion.div
          className="h-full origin-left bg-neutral-700"
          initial={reduceMotion ? false : { scaleX: 0 }}
          animate={{ scaleX: Math.max(0, Math.min(hit.rerank_score, 1)) }}
          transition={{ duration: reduceMotion ? 0 : 0.38, delay: reduceMotion ? 0 : Math.min(index, 5) * 0.045 }}
        />
      </div>

      <p className="line-clamp-5 break-words text-[13px] leading-relaxed text-neutral-600">{hit.snippet}</p>

      <button
        type="button"
        onClick={() => onOpen(hit)}
        className="mt-3 inline-flex items-center gap-1.5 text-[11.5px] font-medium text-neutral-500 transition-colors hover:text-neutral-900 focus-visible:rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-400"
      >
        <FileSearch className="size-3.5" />
        Open document
      </button>
    </motion.article>
  )
}

export function EvidenceRail({
  run,
  highlight,
  onOpen,
}: {
  run: RunState | null
  highlight: { n: number; nonce: number } | null
  onOpen: (hit: Hit) => void
}) {
  const list = useRef<HTMLDivElement>(null)
  const rerank = run?.events.rerank
  const fuse = run?.events.fuse
  const cited = new Set(run?.events.generate?.citations.map((c) => c.n) ?? [])

  useEffect(() => {
    if (!highlight) return
    list.current
      ?.querySelector(`[data-rank="${highlight.n}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" })
  }, [highlight])

  const label = run ? progressLabel(run) : null

  return (
    <div className="flex flex-col gap-3">
      {run && (
        <p className="px-1 text-[12px] leading-relaxed text-neutral-500">
          Sources for: <span className="text-neutral-700">{run.question}</span>
        </p>
      )}
      <div className="flex items-baseline justify-between gap-2 px-1">
        <h2 className="text-[12px] font-semibold uppercase tracking-[0.11em] text-neutral-500">Evidence</h2>
        {rerank && (
          <span className="text-[11px] font-medium text-neutral-400">
            {rerank.hits.length} found · {cited.size} cited
          </span>
        )}
      </div>

      {rerank ? (
        <div ref={list} className="flex flex-col gap-3" data-role="hits">
          {rerank.hits.map((hit, i) => (
            <EvidenceCard
              key={hit.doc_id}
              hit={hit}
              index={i}
              cited={cited.has(hit.rank)}
              highlighted={highlight?.n === hit.rank}
              onOpen={onOpen}
            />
          ))}
        </div>
      ) : run && !run.outcome ? (
        <div className="flex flex-col gap-3 px-1 py-2">
          {label && (
            <div className="text-[13px] text-neutral-500">
              <TextShimmer duration={2.4} className="motion-reduce:animate-none">
                {label}
              </TextShimmer>
            </div>
          )}
          {fuse ? (
            <div className="flex flex-col gap-1.5" data-role="fused-preview">
              <p className="text-[10.5px] font-semibold uppercase tracking-[0.09em] text-neutral-400">
                Top of the fused list, before reranking
              </p>
              {fuse.candidates.slice(0, 10).map((c, i) => (
                <div key={c.doc_id} className="flex items-center gap-2.5 rounded-xl bg-white/70 px-3 py-2 text-[12px]">
                  <span className="w-6 shrink-0 text-right font-semibold tabular-nums text-neutral-400">{i + 1}</span>
                  <span className="min-w-0 flex-1 truncate font-medium text-neutral-700">{c.title}</span>
                  <SourceTag source={c.source} className="shrink-0 text-[10.5px] text-neutral-500" />
                </div>
              ))}
            </div>
          ) : (
            <div className="flex animate-pulse flex-col gap-3" data-role="evidence-skeleton">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="h-28 rounded-[18px] bg-white/70" />
              ))}
            </div>
          )}
        </div>
      ) : (
        <p className="px-4 py-8 text-center text-[13px] leading-relaxed text-neutral-400">
          {run?.outcome
            ? "No documents: the run ended before reranking."
            : "The 10 documents the answer is built from will appear here, with their sources and scores."}
        </p>
      )}
    </div>
  )
}
