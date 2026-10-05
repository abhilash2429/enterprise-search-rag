// Rows for the Retrieval tab: each fused candidate's rank in every retrieval list, its RRF score, and where the
// reranker put it.

import type { Fuse, ListName, Question, Rerank, Source } from "@/lib/api"

export const LISTS: ListName[] = ["bm25", "dense", "bm25_routed", "dense_routed"]

export const LIST_LABEL: Record<ListName, string> = {
  bm25: "BM25",
  dense: "Dense",
  bm25_routed: "BM25 routed",
  dense_routed: "Dense routed",
}

/**
 * The retriever behind each list. A routed list is the same retriever restricted to the routed sources, so "found by
 * one retriever" means found only by BM25 (all or routed) or only by dense search (all or routed).
 */
export const RETRIEVER: Record<ListName, "bm25" | "dense"> = {
  bm25: "bm25",
  dense: "dense",
  bm25_routed: "bm25",
  dense_routed: "dense",
}

export type FoundBy = "both" | "bm25" | "dense"

export type RetrievalRow = {
  fusedRank: number // 1..100
  doc_id: string
  source: Source
  title: string
  ranks: Record<ListName, number | null>
  rrf: number
  rerankRank: number | null // 1..10 when the reranker kept it in the top 10
  foundBy: FoundBy
}

export function retrievalRows(fuse: Fuse, rerank: Rerank | undefined): RetrievalRow[] {
  const top = new Map((rerank?.order ?? []).slice(0, 10).map((o, i) => [o.doc_id, i + 1]))
  return fuse.candidates.map((c, i) => {
    const bm25 = c.ranks.bm25 !== null || c.ranks.bm25_routed !== null
    const dense = c.ranks.dense !== null || c.ranks.dense_routed !== null
    return {
      fusedRank: i + 1,
      doc_id: c.doc_id,
      source: c.source,
      title: c.title,
      ranks: c.ranks,
      rrf: c.rrf_score,
      rerankRank: top.get(c.doc_id) ?? null,
      foundBy: bm25 && dense ? "both" : bm25 ? "bm25" : "dense",
    }
  })
}

export function foundByCounts(rows: RetrievalRow[]): Record<FoundBy, number> {
  const counts: Record<FoundBy, number> = { both: 0, bm25: 0, dense: 0 }
  for (const r of rows) counts[r.foundBy] += 1
  return counts
}

export type GoldDoc =
  | { doc_id: string; status: "top10"; rank: number; title: string }
  | { doc_id: string; status: "missed"; fusedRank: number | null; title: string | null }

/** For each gold document: its rank in the reranked top 10, or missed (with its fused rank if it was a candidate). */
export function goldDocs(question: Question, fuse: Fuse | undefined, rerank: Rerank | undefined): GoldDoc[] {
  return question.expected_doc_ids.map((doc_id) => {
    const hit = rerank?.hits.find((h) => h.doc_id === doc_id)
    if (hit) return { doc_id, status: "top10", rank: hit.rank, title: hit.title }
    const i = fuse?.candidates.findIndex((c) => c.doc_id === doc_id) ?? -1
    return {
      doc_id,
      status: "missed",
      fusedRank: i === -1 ? null : i + 1,
      title: i === -1 ? null : fuse!.candidates[i].title,
    }
  })
}
