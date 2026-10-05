"use client"

import { FileSearch } from "lucide-react"
import { AnimatePresence, motion, useReducedMotion } from "motion/react"
import { useEffect, useRef, useState } from "react"

import { TextShimmer } from "@/components/ui/text-shimmer"
import type { Fuse, Rerank } from "@/lib/api"
import type { RunState } from "@/lib/run"
import { cn } from "@/lib/utils"

import { SourceTag } from "./source-dot"
import { progressLabel } from "./stage-trace"

export type Hit = Rerank["hits"][number]
type Candidate = Fuse["candidates"][number]

/** How long the fused order stays on screen, marked kept or dropped, before cards move to their reranked slots. */
export const RERANK_HOLD_MS = 900

const LAYOUT = { type: "spring", stiffness: 260, damping: 30, mass: 0.9 } as const

function HitCard({
  hit,
  cited,
  highlighted,
  promotedFrom,
  onOpen,
}: {
  hit: Hit
  cited: boolean
  highlighted: boolean
  promotedFrom: number | null
  onOpen: (hit: Hit) => void
}) {
  return (
    <div
      data-rank={hit.rank}
      data-doc={hit.doc_id}
      data-cited={cited}
      data-highlighted={highlighted}
      aria-label={`Rank ${hit.rank}: ${hit.title}${cited ? ", cited" : ""}`}
      role="group"
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
            cited ? "bg-neutral-800 text-white" : "bg-neutral-100 text-neutral-700"
          )}
          aria-hidden
        >
          {hit.rank}
        </span>
        <div className="min-w-0 flex-1">
          <p className="line-clamp-2 text-[12.5px] font-semibold leading-snug text-neutral-800">{hit.title}</p>
          <p className="mt-1 flex flex-wrap items-center gap-x-1.5 text-[10.5px] font-medium uppercase tracking-[0.08em] text-neutral-600">
            <SourceTag source={hit.source} className="text-neutral-700" />
            <span>· score {hit.rerank_score.toFixed(3)}</span>
            <span>· was #{hit.fused_rank} before reranking</span>
            {cited && <span className="font-bold text-neutral-800">· cited</span>}
          </p>
        </div>
        {promotedFrom !== null && (
          <span className="shrink-0 rounded-full bg-neutral-800 px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.05em] text-white">
            up from #{promotedFrom}
          </span>
        )}
      </div>

      <div className="mb-3 h-px overflow-hidden bg-neutral-100">
        <div className="h-full origin-left bg-neutral-700" style={{ transform: `scaleX(${Math.max(0, Math.min(hit.rerank_score, 1))})` }} />
      </div>

      <p className="line-clamp-5 break-words text-[13px] leading-relaxed text-neutral-700">{hit.snippet}</p>

      <button
        type="button"
        onClick={() => onOpen(hit)}
        aria-label={`Open document ${hit.rank}: ${hit.title}`}
        className="mt-3 inline-flex items-center gap-1.5 rounded text-[11.5px] font-medium text-neutral-600 transition-colors hover:text-neutral-900"
      >
        <FileSearch className="size-3.5" aria-hidden />
        Open document
      </button>
    </div>
  )
}

function FusedCard({ c, fusedRank, verdict }: { c: Candidate; fusedRank: number; verdict: number | "dropped" | null }) {
  return (
    <div
      data-fused-rank={fusedRank}
      data-doc={c.doc_id}
      data-verdict={verdict === null ? undefined : verdict === "dropped" ? "dropped" : "kept"}
      className={cn(
        "flex items-start gap-3 rounded-[18px] bg-white/80 p-3.5 transition-opacity duration-300",
        verdict === "dropped" && "opacity-45"
      )}
    >
      <span
        className="flex size-7 shrink-0 items-center justify-center rounded-full bg-neutral-100 text-[11px] font-semibold tabular-nums text-neutral-700 ring-1 ring-inset ring-neutral-300"
        aria-hidden
      >
        {fusedRank}
      </span>
      <div className="min-w-0 flex-1">
        <p className="line-clamp-2 text-[12.5px] font-semibold leading-snug text-neutral-800">{c.title}</p>
        <p className="mt-1 flex flex-wrap items-center gap-x-1.5 text-[10.5px] font-medium uppercase tracking-[0.08em] text-neutral-600">
          <SourceTag source={c.source} className="text-neutral-700" />
          <span>· fused #{fusedRank}</span>
          <span>· rrf {c.rrf_score.toFixed(4)}</span>
        </p>
      </div>
      {verdict !== null && (
        <span
          className={cn(
            "shrink-0 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-[0.05em]",
            verdict === "dropped" ? "bg-neutral-200 text-neutral-800" : "bg-neutral-800 text-white"
          )}
        >
          {verdict === "dropped" ? "dropped" : `kept, now #${verdict}`}
        </span>
      )}
    </div>
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
  const reduceMotion = useReducedMotion() === true
  const rerank = run?.events.rerank
  const fuse = run?.events.fuse
  const cited = new Set(run?.events.generate?.citations.map((c) => c.n) ?? [])
  // A rail mounted after rerank arrived (an older turn) shows the reranked order at once; one that sees rerank
  // arrive plays the reorder. The parent remounts the rail per turn.
  const hasRerank = rerank !== undefined
  const [phase, setPhase] = useState<"fused" | "judging" | "reranked">(hasRerank ? "reranked" : "fused")
  const mountedWithRerank = useRef(hasRerank)

  // Plays once, when rerank arrives: hold the fused order marked kept or dropped, then move to the reranked order.
  // The effect must not depend on `phase`, or its cleanup would cancel the settle timer when the phase changes.
  useEffect(() => {
    if (!hasRerank || mountedWithRerank.current) return
    const judge = setTimeout(() => setPhase(reduceMotion ? "reranked" : "judging"), 0)
    const settle = setTimeout(() => setPhase("reranked"), reduceMotion ? 0 : RERANK_HOLD_MS)
    return () => {
      clearTimeout(judge)
      clearTimeout(settle)
    }
  }, [hasRerank, reduceMotion])

  useEffect(() => {
    if (!highlight) return
    list.current?.querySelector(`[data-rank="${highlight.n}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" })
  }, [highlight])

  const label = run ? progressLabel(run) : null
  const showReranked = rerank !== undefined && phase === "reranked"
  const fusedTop = fuse?.candidates.slice(0, 10) ?? []
  const newRank = new Map(rerank?.hits.map((h) => [h.doc_id, h.rank]) ?? [])

  let status: React.ReactNode = null
  if (showReranked) {
    status = `${rerank.hits.length} found · ${cited.size} cited`
  } else if (rerank && phase === "judging") {
    const kept = fusedTop.filter((c) => newRank.has(c.doc_id)).length
    status = `Reranker scored all 100: ${kept} of the fused top 10 stay`
  } else if (fuse) {
    status = "Top 10 after fusion, before reranking"
  }

  return (
    <div className="flex flex-col gap-3">
      {run && (
        <p className="px-1 text-[12px] leading-relaxed text-neutral-600">
          Sources for: <span className="text-neutral-800">{run.question}</span>
        </p>
      )}
      <div className="flex items-baseline justify-between gap-2 px-1">
        <h2 className="text-[12px] font-semibold uppercase tracking-[0.11em] text-neutral-600">Evidence</h2>
        {status && (
          <span className="text-[11px] font-medium text-neutral-600" aria-live="polite" data-role="evidence-status">
            {status}
          </span>
        )}
      </div>

      {fuse ? (
        <>
          {!showReranked && label && !rerank && (
            <div className="px-1 text-[13px] text-neutral-600">
              <TextShimmer duration={2.4} className="motion-reduce:animate-none">
                {label}
              </TextShimmer>
            </div>
          )}
          <div
            ref={list}
            className="flex flex-col gap-3"
            data-role={showReranked ? "hits" : "fused-preview"}
            data-phase={phase}
          >
            <AnimatePresence mode="popLayout" initial={false}>
              {showReranked
                ? rerank.hits.map((hit) => (
                    <motion.div
                      key={hit.doc_id}
                      layout={!reduceMotion}
                      initial={reduceMotion ? false : { opacity: 0, y: 28 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ layout: LAYOUT, opacity: { duration: 0.3 }, y: LAYOUT }}
                    >
                      <HitCard
                        hit={hit}
                        cited={cited.has(hit.rank)}
                        highlighted={highlight?.n === hit.rank}
                        promotedFrom={hit.fused_rank > 10 ? hit.fused_rank : null}
                        onOpen={onOpen}
                      />
                    </motion.div>
                  ))
                : fusedTop.map((c, i) => (
                    <motion.div
                      key={c.doc_id}
                      layout={!reduceMotion}
                      initial={reduceMotion ? false : { opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={reduceMotion ? undefined : { opacity: 0, x: 24, transition: { duration: 0.25 } }}
                      transition={{ layout: LAYOUT, duration: 0.2, delay: Math.min(i, 9) * 0.03 }}
                    >
                      <FusedCard
                        c={c}
                        fusedRank={i + 1}
                        verdict={rerank && phase === "judging" ? (newRank.get(c.doc_id) ?? "dropped") : null}
                      />
                    </motion.div>
                  ))}
            </AnimatePresence>
          </div>
        </>
      ) : run && !run.outcome ? (
        <div className="flex flex-col gap-3 px-1 py-2">
          {label && (
            <div className="text-[13px] text-neutral-600">
              <TextShimmer duration={2.4} className="motion-reduce:animate-none">
                {label}
              </TextShimmer>
            </div>
          )}
          <div className="flex animate-pulse flex-col gap-3" data-role="evidence-skeleton">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="h-28 rounded-[18px] bg-white/70" />
            ))}
          </div>
        </div>
      ) : (
        <p className="px-4 py-8 text-center text-[13px] leading-relaxed text-neutral-600">
          {run?.outcome
            ? "No documents: the run ended before fusion."
            : "The 10 documents the answer is built from will appear here, with their sources and scores."}
        </p>
      )}
    </div>
  )
}
