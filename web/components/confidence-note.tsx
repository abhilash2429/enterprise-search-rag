"use client"

import { CircleCheck, TriangleAlert } from "lucide-react"

import type { Confidence } from "@/lib/api"
import { VERDICT_LABEL } from "@/lib/format"
import type { RunState } from "@/lib/run"

import { CitationChips } from "./answer-body"

const ADVISORY =
  "This flag is advisory and never changes the answer. On held-out questions, unflagged answers were right about 90% of the time and flagged ones about 66%."

function LowRetrieval({ c }: { c: Confidence }) {
  if (!c.low_retrieval_score) return null
  const top = c.rerank_top !== null ? c.rerank_top.toFixed(3) : "unknown"
  const threshold = c.threshold !== undefined ? c.threshold.toFixed(3) : "the threshold"
  return (
    <p data-role="low-retrieval">
      <span className="font-semibold">Retrieval confidence was low:</span> top reranker score {top}, below the{" "}
      {threshold} threshold.
    </p>
  )
}

export function ConfidenceNote({
  run,
  active,
  onCite,
  titles,
}: {
  run: RunState
  active: number | null
  onCite: (n: number) => void
  titles: Record<number, string>
}) {
  const stages = run.events.start?.stages ?? []
  const v = run.events.verify
  if (!run.events.generate || !stages.includes("verify")) return null

  if (!v) {
    if (run.outcome) return null
    return (
      <div className="flex animate-pulse flex-col gap-2" data-role="verify-skeleton">
        <div className="h-3 w-2/3 rounded bg-neutral-100" />
        <div className="h-3 w-1/2 rounded bg-neutral-100" />
      </div>
    )
  }
  if (v.skipped === "abstained" || !v.confidence) {
    return <p className="text-[12.5px] text-neutral-600">Not verified: refusals are not checked.</p>
  }
  const c = v.confidence
  if (c.flagged === null) {
    return (
      <div className="rounded-xl bg-neutral-100 px-3.5 py-3 text-[13px] leading-relaxed text-neutral-700" data-banner="unparsed">
        <p className="font-semibold">Confidence unknown</p>
        <p>The verifier reply did not parse{c.error ? `: ${c.error}` : "."}</p>
        <LowRetrieval c={c} />
      </div>
    )
  }
  if (!c.flagged) {
    return (
      <div className="flex flex-col gap-1 text-[12.5px] text-neutral-600" data-banner="verified">
        <p className="flex items-center gap-1.5 font-medium text-neutral-700">
          <CircleCheck className="size-4" aria-hidden />
          Verified against the cited documents
        </p>
        <div className="pl-[22px]">
          <LowRetrieval c={c} />
          <p>The verifier found the cited claims supported. Advisory only; the answer is unchanged.</p>
        </div>
      </div>
    )
  }

  const doubtful = (c.claims ?? []).filter((x) => x.verdict !== "supported")
  return (
    <section
      className="rounded-[18px] bg-amber-50 px-4 py-3.5 text-[13.5px] leading-relaxed text-amber-950 ring-1 ring-inset ring-amber-200"
      data-banner="flagged"
    >
      <p className="flex items-center gap-2 text-[14px] font-semibold">
        <TriangleAlert className="size-4 text-amber-600" aria-hidden />
        Check the cited documents
        {c.verdict && (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10.5px] font-bold uppercase tracking-[0.06em] text-amber-800">
            {VERDICT_LABEL[c.verdict]}
          </span>
        )}
      </p>
      <div className="mt-2 flex flex-col gap-2 pl-6">
        {c.addresses_question === false && <p>The verifier says the answer does not address the question.</p>}
        <LowRetrieval c={c} />
        {c.reasoning && <p data-role="reasoning">{c.reasoning}</p>}
        {doubtful.length > 0 ? (
          <div className="flex flex-col gap-1.5">
            <p className="text-[10.5px] font-semibold uppercase tracking-[0.09em] text-amber-800">
              {doubtful.length} claim{doubtful.length === 1 ? "" : "s"} not fully supported
            </p>
            <ul className="flex flex-col gap-1.5" data-role="doubtful-claims">
              {doubtful.map((x, i) => (
                <li key={i} className="rounded-xl bg-white/70 px-3 py-2">
                  <span className="mr-2 text-[10.5px] font-bold uppercase tracking-[0.06em] text-amber-700">
                    {VERDICT_LABEL[x.verdict]}
                  </span>
                  {x.claim}
                  {x.cited.length > 0 && <CitationChips ns={x.cited} active={active} onCite={onCite} titles={titles} />}
                </li>
              ))}
            </ul>
          </div>
        ) : (
          c.claims && <p>Every claim is supported by its cited document.</p>
        )}
        <p className="text-[12px] text-amber-900/80">{ADVISORY}</p>
      </div>
    </section>
  )
}
