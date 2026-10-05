"use client"

import type { ListName } from "@/lib/api"
import { foundByCounts, LISTS, retrievalRows, RETRIEVER, type FoundBy } from "@/lib/retrieval"
import type { RunState } from "@/lib/run"
import { cn } from "@/lib/utils"

import { SourceTag } from "./source-dot"

const HEADER: Record<ListName, [string, string]> = {
  bm25: ["BM25", "all"],
  dense: ["Dense", "all"],
  bm25_routed: ["BM25", "routed"],
  dense_routed: ["Dense", "routed"],
}

const FOUND_LABEL: Record<FoundBy, string> = { both: "Both", bm25: "BM25 only", dense: "Dense only" }

/** A filled cell means that list returned the document, with its rank; empty means it did not. */
function RankCell({ rank, list }: { rank: number | null; list: ListName }) {
  if (rank === null) return <td className="px-1 py-1" aria-label="not found" />
  return (
    <td className="px-1 py-1">
      <span
        className={cn(
          "inline-flex h-6 min-w-9 items-center justify-center rounded-md px-1 text-[12px] font-semibold tabular-nums",
          list.endsWith("_routed")
            ? RETRIEVER[list] === "bm25" ? "retr-bm25-routed" : "retr-dense-routed"
            : RETRIEVER[list] === "bm25" ? "retr-bm25" : "retr-dense"
        )}
      >
        {rank}
      </span>
    </td>
  )
}

export function RetrievalTable({ run }: { run: RunState | null }) {
  const fuse = run?.events.fuse
  if (!run || !fuse) {
    return (
      <p className="px-4 py-8 text-center text-[13px] leading-relaxed text-neutral-600">
        {run && !run.outcome
          ? "The 100 fused candidates appear here once the Fuse stage finishes."
          : "Ask a question to see how BM25 and dense retrieval combine."}
      </p>
    )
  }
  const rows = retrievalRows(fuse, run.events.rerank)
  const counts = foundByCounts(rows)

  return (
    <div className="flex flex-col gap-3" data-role="retrieval">
      <div className="flex flex-col gap-2 px-1">
        <div className="flex items-baseline justify-between gap-2">
          <h2 className="text-[12px] font-semibold uppercase tracking-[0.11em] text-neutral-600">Retrieval</h2>
          <span className="text-[11px] font-medium text-neutral-600">{rows.length} fused candidates</span>
        </div>
        <p className="text-[12px] leading-relaxed text-neutral-600">
          Rank of each document in the four ranked lists, fused by RRF: the score is the sum of 1/(60 + rank) over the
          lists that found it. Filled chips search everything; outlined chips search only the sources the router picked.
        </p>
        <div className="flex flex-wrap items-center gap-2 text-[12px] font-semibold" data-role="found-by-counts">
          <span className="inline-flex items-center gap-1.5 rounded-full bg-white px-2.5 py-1 text-neutral-800">
            Both retrievers <span className="tabular-nums" data-count="both">{counts.both}</span>
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 retr-only-bm25">
            BM25 only <span className="tabular-nums" data-count="bm25">{counts.bm25}</span>
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 retr-only-dense">
            Dense only <span className="tabular-nums" data-count="dense">{counts.dense}</span>
          </span>
        </div>
      </div>

      <div className="overflow-hidden rounded-[18px] bg-white">
        <table className="w-full table-fixed border-collapse text-left text-[12px]" aria-label="Fused candidates">
          <colgroup>
            <col className="w-9" />
            <col />
            {LISTS.map((l) => (
              <col key={l} className="w-12" />
            ))}
            <col className="w-[4.25rem]" />
            <col className="w-[3.25rem]" />
          </colgroup>
          <thead className="sticky top-0 z-10 bg-neutral-100 text-[10.5px] uppercase tracking-[0.06em] text-neutral-600">
            <tr>
              <th scope="col" className="px-2 py-2 text-right font-semibold" title="Fused rank">
                #
              </th>
              <th scope="col" className="px-2 py-2 font-semibold">
                Document
              </th>
              {LISTS.map((l) => (
                <th key={l} scope="col" className="px-1 py-2 text-center font-semibold leading-tight">
                  {HEADER[l][0]}
                  <span className="block text-[9.5px] font-medium normal-case tracking-normal">{HEADER[l][1]}</span>
                </th>
              ))}
              <th scope="col" className="px-1 py-2 text-right font-semibold">
                RRF
              </th>
              <th scope="col" className="px-2 py-2 text-center font-semibold leading-tight">
                Top 10
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.doc_id}
                data-fused-rank={r.fusedRank}
                data-doc={r.doc_id}
                data-found-by={r.foundBy}
                aria-label={`Fused rank ${r.fusedRank}, ${r.title}, ${FOUND_LABEL[r.foundBy]}`}
                className={cn(
                  "border-t border-neutral-100 align-middle",
                  r.foundBy === "bm25" && "retr-row-bm25",
                  r.foundBy === "dense" && "retr-row-dense"
                )}
              >
                <td className="px-2 py-1 text-right font-semibold tabular-nums text-neutral-600">{r.fusedRank}</td>
                <td className="min-w-0 px-2 py-1">
                  <span className="block truncate font-medium text-neutral-800">{r.title}</span>
                  <span className="flex items-center gap-1.5 text-[10.5px] text-neutral-600">
                    <SourceTag source={r.source} />
                    {r.foundBy !== "both" && <span className="font-bold uppercase tracking-[0.05em]">· {FOUND_LABEL[r.foundBy]}</span>}
                  </span>
                </td>
                {LISTS.map((l) => (
                  <RankCell key={l} rank={r.ranks[l]} list={l} />
                ))}
                <td className="px-1 py-1 text-right font-mono text-[11.5px] tabular-nums text-neutral-700">{r.rrf.toFixed(4)}</td>
                <td className="px-2 py-1 text-center">
                  {r.rerankRank !== null && (
                    <span className="inline-flex size-6 items-center justify-center rounded-full bg-neutral-800 text-[11px] font-bold tabular-nums text-white">
                      {r.rerankRank}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
