"use client"

import { Reasoning, ReasoningContent, ReasoningTrigger } from "@/components/ui/reasoning"
import { formatCost, formatSeconds, SOURCE_LABEL } from "@/lib/format"
import type { RunState } from "@/lib/run"

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="pt-px text-[11px] font-semibold uppercase tracking-[0.03em] text-muted-foreground">{label}</dt>
      <dd className="m-0 text-[13px] leading-snug">{children}</dd>
    </>
  )
}

export function WhyPanel({ run }: { run: RunState }) {
  const { start, route, retrieve, fuse, rerank, generate, verify, done } = run.events
  const lists = retrieve ? Object.entries(retrieve.lists).filter(([, v]) => v !== null) : []
  const conf = verify?.confidence

  return (
    <Reasoning>
      <ReasoningTrigger className="text-[12px] font-medium text-neutral-500 hover:text-neutral-800">
        Why this answer
      </ReasoningTrigger>
      <ReasoningContent className="mt-2" contentClassName="bg-neutral-50 px-3 py-2.5">
        <dl className="grid grid-cols-[108px_1fr] gap-x-3 gap-y-1.5">
          {start && (
            <Row label="Config">
              router {start.config.router ? "on" : "off"}, rerank {start.config.rerank ? "on" : "off"}, dense{" "}
              {start.config.dense}, verify {start.config.verify ? "on" : "off"}
            </Row>
          )}
          {route && (
            <Row label="Route">
              {route.sources.length ? route.sources.map((s) => SOURCE_LABEL[s]).join(", ") : "no source named"}
            </Row>
          )}
          {retrieve && (
            <Row label="Retrieved">
              {lists.map(([name, ids]) => `${name} ${ids?.length ?? 0}`).join(", ")}
            </Row>
          )}
          {fuse && <Row label="Fused">{fuse.candidates.length} candidates by reciprocal rank fusion</Row>}
          {rerank && (
            <Row label="Reranked">
              top {rerank.hits.length} of {rerank.order.length}, best score {rerank.hits[0]?.rerank_score.toFixed(3)}
              {conf?.threshold !== undefined && `, flag threshold ${conf.threshold.toFixed(3)}`}
            </Row>
          )}
          {generate && (
            <Row label="Cited">
              {generate.citations.length
                ? generate.citations.map((c) => `${c.n} ${c.title}`).join("; ")
                : "nothing cited"}
            </Row>
          )}
          {generate && generate.version_pairs.length > 0 && (
            <Row label="Versions">
              {generate.version_pairs.map((p) => `${p.a} and ${p.b} (cosine ${p.cosine.toFixed(3)})`).join(", ")}
            </Row>
          )}
          {verify && (
            <Row label="Verifier">
              {verify.skipped === "abstained"
                ? "skipped for a refusal"
                : conf
                  ? `${conf.flagged === null ? "reply did not parse" : conf.flagged ? "flagged" : "not flagged"}${conf.verdict ? `, ${conf.verdict.replace("_", " ")}` : ""}`
                  : "no result"}
            </Row>
          )}
          {done && (
            <Row label="Time">
              {Object.entries(done.seconds)
                .filter(([k]) => k !== "total")
                .map(([k, s]) => `${k} ${formatSeconds(s ?? 0)}`)
                .join(", ")}
              ; total {formatSeconds(done.seconds.total)}, {formatCost(done.cost_usd)}
            </Row>
          )}
        </dl>
      </ReasoningContent>
    </Reasoning>
  )
}
