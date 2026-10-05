"use client"

import { ChevronRight } from "lucide-react"
import { AnimatePresence, motion, useReducedMotion } from "motion/react"
import { useId } from "react"

import { TextShimmer } from "@/components/ui/text-shimmer"
import type { Stage } from "@/lib/api"
import { formatCost, formatSeconds } from "@/lib/format"
import { useNow } from "@/lib/hooks"
import { runningStage, stageViews, type RunState, type StageView } from "@/lib/run"
import { cn } from "@/lib/utils"

import { SourceTag } from "./source-dot"

export const STAGE_LABEL: Record<Stage, string> = {
  route: "Route",
  retrieve: "Retrieve",
  fuse: "Fuse",
  rerank: "Rerank",
  generate: "Generate",
  verify: "Verify",
}

/** Pipeline statuses, not a transcript of the model's reasoning. */
const PROGRESS: Record<Stage, string> = {
  route: "Choosing which sources to search",
  retrieve: "Searching 512K documents with BM25 and dense retrieval",
  fuse: "Fusing the ranked lists",
  rerank: "Reranking 100 candidates",
  generate: "Writing a cited answer from the top 10",
  verify: "Checking each cited claim against its document",
}

export function progressLabel(run: RunState): string | null {
  if (run.phase === "checking") return "Checking the backend"
  const stage = runningStage(run)
  return stage ? PROGRESS[stage] : null
}

/** One line on what a finished stage produced, from its own event. */
function stageSummary(run: RunState, stage: Stage): React.ReactNode {
  const e = run.events
  switch (stage) {
    case "route": {
      const sources = e.route?.sources ?? []
      return (
        <span className="flex flex-wrap items-center gap-x-3 gap-y-1" data-role="routed">
          <span>Routed to</span>
          {sources.length > 0 ? (
            sources.map((s) => <SourceTag key={s} source={s} className="text-neutral-700" />)
          ) : (
            <span>no source named; searching everything</span>
          )}
        </span>
      )
    }
    case "retrieve": {
      const routed = e.retrieve?.lists.bm25_routed !== null || e.retrieve?.lists.dense_routed !== null
      return `BM25 and dense over every source${routed ? ", and both again over the routed sources" : ""}`
    }
    case "fuse":
      return `${e.fuse?.candidates.length ?? 0} candidates by reciprocal rank fusion`
    case "rerank":
      return `Kept the top ${e.rerank?.hits.length ?? 0} of ${e.fuse?.candidates.length ?? 0}`
    case "generate": {
      const g = e.generate
      if (!g) return null
      if (g.abstained) return "Declined: the documents do not answer it"
      const n = g.citations.length
      return `${g.partial ? "Partial answer" : "Answer"} citing ${n} document${n === 1 ? "" : "s"}`
    }
    case "verify": {
      const v = e.verify
      if (!v) return null
      if (v.skipped) return "Skipped: refusals are not checked"
      if (v.confidence?.flagged === true) return "Flagged for a second look"
      if (v.confidence?.flagged === false) return "Cited claims supported"
      return "Verifier reply did not parse"
    }
  }
}

function stageAria(v: StageView): string {
  const state =
    v.status === "done" && v.seconds !== null
      ? `done in ${formatSeconds(v.seconds)}`
      : v.status === "running"
        ? "running"
        : v.status
  return `${STAGE_LABEL[v.stage]}: ${state}`
}

function Marker({ status }: { status: StageView["status"] }) {
  return (
    <span className="relative z-10 flex size-4 items-center justify-center bg-white" aria-hidden>
      {status === "running" && (
        <span className="absolute size-2.5 animate-ping rounded-full bg-neutral-400 opacity-60 motion-reduce:animate-none" />
      )}
      <span
        className={cn(
          "relative size-2 rounded-full",
          status === "done" && "bg-neutral-400",
          status === "running" && "bg-neutral-800",
          (status === "failed" || status === "stopped") && "bg-white ring-2 ring-neutral-800"
        )}
      />
    </span>
  )
}

/**
 * The pipeline as a run of steps. Collapsed, one line names the step in progress and changes as each step starts;
 * expanded, the steps stack vertically as they start, each with what it produced and its time.
 */
export function StageTrace({ run, open, onToggle }: { run: RunState; open: boolean; onToggle: () => void }) {
  const active = run.phase === "checking" || run.phase === "streaming"
  const now = useNow(active)
  const reduceMotion = useReducedMotion()
  const id = useId()
  const views = stageViews(run)
  const shown = views.filter((v) => v.status !== "pending")
  const done = run.events.done
  const label = progressLabel(run)
  const live = active && now > 0 ? Math.max(0, now - run.startedAt) / 1000 : null
  const ended = views.find((v) => v.status === "failed" || v.status === "stopped")
  // A run turned away before any step (busy, loading, unknown question) shows only its failure card.
  if (!active && !done && shown.length === 0) return null

  const header = active
    ? label
    : done
      ? `Ran ${views.length} steps`
      : ended
        ? `Stopped at ${STAGE_LABEL[ended.stage]}`
        : `Stopped after ${shown.length} steps`

  return (
    <div className="flex flex-col gap-2" data-role="steps">
      <div className="flex min-w-0 items-center gap-3 text-[13px]">
        <button
          type="button"
          aria-expanded={open}
          aria-controls={id}
          onClick={onToggle}
          data-role="steps-toggle"
          className="group inline-flex min-w-0 items-center gap-1 rounded text-left font-medium text-neutral-600 hover:text-neutral-900"
        >
          <span className="min-w-0 overflow-hidden" role="status" aria-live="polite" data-role="steps-label">
            {/* Keyed by the text, so each new step's line slides in as the previous one is replaced. */}
            <motion.span
              key={header ?? ""}
              className="block truncate"
              initial={reduceMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.22, ease: "easeOut" }}
            >
              {active ? (
                <TextShimmer duration={2.4} className="motion-reduce:animate-none">
                  {header}
                </TextShimmer>
              ) : (
                header
              )}
            </motion.span>
          </span>
          <ChevronRight
            className={cn("size-4 shrink-0 transition-transform motion-reduce:transition-none", open && "rotate-90")}
            aria-hidden
          />
        </button>
        <span
          className="ml-auto shrink-0 whitespace-nowrap text-[12px] font-medium tabular-nums text-muted-foreground"
          data-role="total"
        >
          {done ? (
            <>
              {formatSeconds(done.seconds.total)} <span data-role="cost">· {formatCost(done.cost_usd)}</span>
            </>
          ) : live !== null ? (
            formatSeconds(live)
          ) : null}
        </span>
      </div>

      {open && (
        <ol id={id} aria-label="Pipeline stages" className="flex flex-col" data-role="steps-list">
          <AnimatePresence initial={false}>
            {shown.map((v, i) => (
              <motion.li
                key={v.stage}
                data-stage={v.stage}
                data-status={v.status}
                aria-label={stageAria(v)}
                className="relative grid grid-cols-[16px_minmax(0,1fr)_auto] gap-x-2.5 pb-2.5 last:pb-0"
                initial={reduceMotion ? false : { opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, ease: "easeOut" }}
              >
                {i < shown.length - 1 && (
                  <span className="absolute bottom-0 left-[7.5px] top-[18px] w-px bg-neutral-200" aria-hidden />
                )}
                <span className="pt-[3px]">
                  <Marker status={v.status} />
                </span>
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="text-[13px] font-semibold text-neutral-800">{STAGE_LABEL[v.stage]}</span>
                  <span className="text-[12.5px] text-neutral-600">
                    {v.status === "done" ? (
                      stageSummary(run, v.stage)
                    ) : v.status === "running" ? (
                      <TextShimmer duration={2.4} className="motion-reduce:animate-none">
                        {PROGRESS[v.stage]}
                      </TextShimmer>
                    ) : v.status === "failed" ? (
                      "Failed"
                    ) : (
                      "Stopped"
                    )}
                  </span>
                </div>
                <span className="pt-px text-[12px] font-medium tabular-nums text-neutral-600" data-role="stage-time">
                  {v.status === "done" && v.seconds !== null && formatSeconds(v.seconds)}
                  {v.status === "running" && v.since !== null && formatSeconds(Math.max(0, now - v.since) / 1000)}
                </span>
              </motion.li>
            ))}
          </AnimatePresence>
        </ol>
      )}
    </div>
  )
}
