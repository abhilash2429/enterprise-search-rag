"use client"

import { Check, Circle, LoaderCircle, X } from "lucide-react"

import { TextShimmer } from "@/components/ui/text-shimmer"
import type { Stage } from "@/lib/api"
import { formatCost, formatSeconds } from "@/lib/format"
import { useNow } from "@/lib/hooks"
import { runningStage, stageViews, type RunState } from "@/lib/run"
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

export function StageTrace({ run }: { run: RunState }) {
  const active = run.phase === "checking" || run.phase === "streaming"
  const now = useNow(active)
  const views = stageViews(run)
  const done = run.events.done
  const sources = run.events.route?.sources
  const label = progressLabel(run)
  const live = active && now > 0 ? Math.max(0, now - run.startedAt) / 1000 : null

  return (
    <div className="flex flex-col gap-2">
      <ol className="flex flex-wrap items-center gap-1.5" aria-label="Pipeline">
        {views.map((v) => (
          <li
            key={v.stage}
            data-stage={v.stage}
            data-status={v.status}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[12px] font-medium transition-colors",
              v.status === "done" && "bg-neutral-100 text-neutral-700",
              v.status === "running" && "bg-neutral-800 text-white",
              v.status === "pending" && "text-neutral-400 ring-1 ring-inset ring-neutral-200",
              v.status === "stopped" && "text-neutral-400 ring-1 ring-inset ring-neutral-200",
              v.status === "failed" && "bg-neutral-200 text-neutral-900"
            )}
          >
            {v.status === "done" && <Check className="size-3.5" aria-hidden />}
            {v.status === "running" && <LoaderCircle className="size-3.5 animate-spin" aria-hidden />}
            {(v.status === "pending" || v.status === "stopped") && <Circle className="size-3" aria-hidden />}
            {v.status === "failed" && <X className="size-3.5" aria-hidden />}
            {STAGE_LABEL[v.stage]}
            <span className="tabular-nums opacity-80" data-role="stage-time">
              {v.status === "done" && v.seconds !== null && formatSeconds(v.seconds)}
              {v.status === "running" && v.since !== null && formatSeconds(Math.max(0, now - v.since) / 1000)}
            </span>
          </li>
        ))}
        <li
          className="ml-1 text-[12px] font-medium tabular-nums text-muted-foreground"
          data-role="total"
        >
          {done ? (
            <>
              {formatSeconds(done.seconds.total)} <span data-role="cost">· {formatCost(done.cost_usd)}</span>
            </>
          ) : live !== null ? (
            formatSeconds(live)
          ) : null}
        </li>
      </ol>
      {sources && (
        <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] font-medium text-neutral-500" data-role="routed">
          <span className="text-[10.5px] font-semibold uppercase tracking-[0.09em] text-neutral-400">Routed to</span>
          {sources.length > 0 ? (
            sources.map((s) => <SourceTag key={s} source={s} className="text-neutral-700" />)
          ) : (
            <span>no source named; searching everything</span>
          )}
        </p>
      )}
      {active && label && (
        <div role="status" aria-live="polite" className="py-0.5 text-[13px] text-neutral-500">
          <TextShimmer duration={2.4} className="motion-reduce:animate-none">
            {label}
          </TextShimmer>
        </div>
      )}
    </div>
  )
}
