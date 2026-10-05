"use client"

import { ChevronDown, ChevronUp, DatabaseZap } from "lucide-react"
import { useState } from "react"

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import type { Health } from "@/lib/api"

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="font-semibold text-slate-500">{label}</dt>
      <dd className="m-0 min-w-0 break-words font-medium text-slate-800">{children}</dd>
    </>
  )
}

export function PipelinePanel({ health, base, mock }: { health: Health | null; base: string; mock: boolean }) {
  const [open, setOpen] = useState(false)
  const c = health?.config
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className="overflow-hidden rounded-xl bg-neutral-200/75">
        <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-3 text-left hover:bg-neutral-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-neutral-500/40">
          <DatabaseZap className="size-4 text-neutral-500" />
          <span className="flex-1 text-[11px] font-semibold uppercase tracking-[0.09em] text-neutral-600">
            Pipeline
          </span>
          {open ? <ChevronUp className="size-4 text-slate-400" /> : <ChevronDown className="size-4 text-slate-400" />}
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="px-3 pb-3 pt-1">
            <dl className="grid grid-cols-[112px_minmax(0,1fr)] gap-x-3 gap-y-2 text-[11.5px] leading-snug">
              <Row label="Mode">{mock ? "Recorded replay of real runs" : "Live backend"}</Row>
              <Row label="API">{base}</Row>
              <Row label="Status">{health ? `${health.status}${health.busy ? ", busy" : ""}` : "unavailable"}</Row>
              {c && (
                <>
                  <Row label="Router">{c.router ? "gpt-oss-120b picks 1 to 3 sources" : "off"}</Row>
                  <Row label="Retrieval">BM25 + Qwen3-Embedding-0.6B ({c.dense}), RRF top 100</Row>
                  <Row label="Reranker">{c.rerank ? "Qwen3-Reranker-0.6B, top 10" : "off"}</Row>
                  <Row label="Answerer">gpt-oss-120b, cited</Row>
                  <Row label="Verifier">{c.verify ? "gpt-6-luna claim check, flag only" : "off"}</Row>
                </>
              )}
              <Row label="Corpus">511,958 documents from 9 sources</Row>
            </dl>
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  )
}
