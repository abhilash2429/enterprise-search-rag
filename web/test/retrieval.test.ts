import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { parseAskEvent, parseQuestions, type Fuse, type Rerank } from "@/lib/api";
import { FIXTURES_DIR } from "@/lib/mock";
import { foundByCounts, goldDocs, LISTS, retrievalRows } from "@/lib/retrieval";

const questions = parseQuestions(JSON.parse(readFileSync(path.join(FIXTURES_DIR, "questions.json"), "utf8")));
const load = (id: string) => {
  const events = readFileSync(path.join(FIXTURES_DIR, "ask", `${id}.jsonl`), "utf8")
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l))
    .map((e) => parseAskEvent(e.event, e.data));
  const get = <T,>(name: string) => events.find((e) => e.event === name)!.data as T;
  return { fuse: get<Fuse>("fuse"), rerank: get<Rerank>("rerank") };
};

describe.each(questions.map((q) => [q.question_id, q] as const))("%s", (id, q) => {
  const { fuse, rerank } = load(id);
  const rows = retrievalRows(fuse, rerank);

  it("has one row per fused candidate with ranks, RRF and reranked position", () => {
    expect(rows).toHaveLength(100);
    expect(rows.map((r) => r.fusedRank)).toEqual(Array.from({ length: 100 }, (_, i) => i + 1));
    for (const r of rows) {
      const rrf = LISTS.reduce((s, l) => s + (r.ranks[l] === null ? 0 : 1 / (60 + r.ranks[l]!)), 0);
      expect(r.rrf).toBeCloseTo(rrf, 12);
      expect(LISTS.some((l) => r.ranks[l] !== null)).toBe(true);
    }
    const reranked = rows.filter((r) => r.rerankRank !== null);
    expect(reranked.map((r) => r.rerankRank).sort((a, b) => a! - b!)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    for (const h of rerank.hits) expect(rows.find((r) => r.doc_id === h.doc_id)?.rerankRank).toBe(h.rank);
    for (const h of rerank.hits) expect(rows[h.fused_rank - 1].doc_id).toBe(h.doc_id);
  });

  it("classifies who found each doc by retriever", () => {
    const counts = foundByCounts(rows);
    expect(counts.both + counts.bm25 + counts.dense).toBe(100);
    for (const r of rows) {
      const bm25 = r.ranks.bm25 !== null || r.ranks.bm25_routed !== null;
      const dense = r.ranks.dense !== null || r.ranks.dense_routed !== null;
      expect(r.foundBy).toBe(bm25 && dense ? "both" : bm25 ? "bm25" : "dense");
    }
  });

  it("places every gold document", () => {
    const gold = goldDocs(q, fuse, rerank);
    expect(gold.map((g) => g.doc_id)).toEqual(q.expected_doc_ids);
    for (const g of gold) {
      const hit = rerank.hits.find((h) => h.doc_id === g.doc_id);
      if (hit) expect(g).toEqual({ doc_id: g.doc_id, status: "top10", rank: hit.rank, title: hit.title });
      else expect(g.status).toBe("missed");
    }
  });
});

it("reports a gold doc outside the 100 candidates as missed with no fused rank", () => {
  const { fuse, rerank } = load("qst_0147");
  const q = { ...questions[0], expected_doc_ids: ["dsid_not_retrieved"] };
  expect(goldDocs(q, fuse, rerank)).toEqual([{ doc_id: "dsid_not_retrieved", status: "missed", fusedRank: null, title: null }]);
});

it("reranked column is blank before rerank arrives", () => {
  const { fuse } = load("qst_0147");
  expect(retrievalRows(fuse, undefined).every((r) => r.rerankRank === null)).toBe(true);
});
