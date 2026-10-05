"use client"

import { LoaderCircle, RotateCw } from "lucide-react"

import type { AskOutcome, Health } from "@/lib/api"

type Failure = Exclude<AskOutcome, { kind: "done" }>

export function failureCopy(outcome: Failure, health: Health | null): { title: string; detail: string; kind: string } {
  if (outcome.kind === "rejected") {
    switch (outcome.reason) {
      case "loading":
        return {
          kind: "loading",
          title: "Loading indexes",
          detail:
            health?.status === "ready"
              ? "The indexes are ready. Running the question now."
              : "The backend is loading its indexes, which takes about a minute after start. Checking again every 2 seconds; the question runs as soon as they are ready.",
        }
      case "busy":
        return {
          kind: "busy",
          title: "Another question is running",
          detail: "The backend answers one question at a time on its GPU. Retry when the current one finishes.",
        }
      case "load_error":
        return { kind: "load_error", title: "The backend failed to load", detail: outcome.message }
      case "health_failed":
        return { kind: "unreachable", title: "The backend is unreachable", detail: outcome.message }
      case "empty":
        return { kind: "empty", title: "Type a question first", detail: "" }
    }
  }
  switch (outcome.kind) {
    case "error":
      return { kind: "error", title: "The pipeline reported an error", detail: outcome.message }
    case "disconnected":
      return { kind: "disconnected", title: "The connection dropped", detail: outcome.message }
    case "invalid":
      return { kind: "invalid", title: "The backend sent a response this UI does not understand", detail: outcome.message }
    case "aborted":
      return { kind: "aborted", title: "Stopped", detail: "The question was stopped before it finished." }
  }
}

export function RunFailure({
  outcome,
  health,
  onRetry,
}: {
  outcome: Failure
  health: Health | null
  onRetry: () => void
}) {
  const copy = failureCopy(outcome, health)
  const loading = copy.kind === "loading"
  return (
    <div
      role="alert"
      data-banner="failure"
      data-failure={copy.kind}
      className="flex items-start gap-3 rounded-xl bg-neutral-100 px-3.5 py-3 text-[13px] leading-relaxed text-neutral-800"
    >
      {loading && <LoaderCircle className="mt-0.5 size-4 shrink-0 animate-spin" aria-hidden />}
      <div className="min-w-0 flex-1">
        <p className="font-semibold text-neutral-900">{copy.title}</p>
        {copy.detail && <p className="mt-0.5 break-words text-neutral-700">{copy.detail}</p>}
      </div>
      {copy.kind !== "empty" && (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-full bg-neutral-800 px-3 py-1.5 text-[12px] font-semibold text-white hover:bg-neutral-700"
        >
          <RotateCw className="size-3.5" aria-hidden />
          {loading ? "Retry now" : "Retry"}
        </button>
      )}
    </div>
  )
}
