// Splits a generated answer into text and citation groups for display.
//
// The contract documents [n], [n][m] and [n, m]. The recorded answers also use fullwidth brackets (【2】), and the
// backend's own parser (`_CITE` in src/entsearch/answerer.py) accepts those plus ";" separators and a "†..." suffix,
// so this mirrors that regex.

export const MISSING_PREFIX = "Not covered by the documents:";

const CITE = /[\[【](\d+(?:\s*[,;]\s*\d+)*)(?:†[^\]】]*)?[\]】]/g;

export type AnswerSegment =
  | { kind: "text"; text: string; notCovered: boolean }
  | { kind: "cite"; ns: number[]; raw: string; notCovered: boolean };

/**
 * Parses `answer` into segments. Adjacent markers ("[2][5][6]") form one group with duplicates dropped. Numbers
 * outside 1..contextSize are not citations: a group with none in range stays as plain text. The sentence starting
 * with "Not covered by the documents:" (up to the end of its line) is marked `notCovered`.
 */
export function parseAnswer(answer: string, contextSize: number): AnswerSegment[] {
  const p = answer.indexOf(MISSING_PREFIX);
  if (p === -1) return parseRange(answer, contextSize, false);
  const nl = answer.indexOf("\n", p);
  const e = nl === -1 ? answer.length : nl;
  return [
    ...parseRange(answer.slice(0, p), contextSize, false),
    ...parseRange(answer.slice(p, e), contextSize, true),
    ...parseRange(answer.slice(e), contextSize, false),
  ];
}

function parseRange(text: string, contextSize: number, notCovered: boolean): AnswerSegment[] {
  const out: AnswerSegment[] = [];
  const pushText = (t: string) => {
    if (!t) return;
    const last = out.at(-1);
    if (last?.kind === "text") last.text += t;
    else out.push({ kind: "text", text: t, notCovered });
  };
  let at = 0;
  for (const m of text.matchAll(CITE)) {
    const between = text.slice(at, m.index);
    const ns = m[1]
      .split(/[,;]/)
      .map((s) => Number(s.trim()))
      .filter((n) => Number.isInteger(n) && n >= 1 && n <= contextSize);
    at = m.index + m[0].length;
    if (ns.length === 0) {
      pushText(between + m[0]);
      continue;
    }
    const last = out.at(-1);
    if (between === "" && last?.kind === "cite") {
      for (const n of ns) if (!last.ns.includes(n)) last.ns.push(n);
      last.raw += m[0];
    } else {
      pushText(between);
      out.push({ kind: "cite", ns: [...new Set(ns)], raw: m[0], notCovered });
    }
  }
  pushText(text.slice(at));
  return out;
}

/** Cited context numbers in order of first citation. */
export function citedOrder(segments: AnswerSegment[]): number[] {
  const seen: number[] = [];
  for (const s of segments) if (s.kind === "cite") for (const n of s.ns) if (!seen.includes(n)) seen.push(n);
  return seen;
}

const toMarkdown = (segments: AnswerSegment[]) =>
  segments.map((s) => (s.kind === "text" ? s.text : s.ns.map((n) => `[${n}](#cite-${n})`).join(""))).join("");

/**
 * Prepares an answer for markdown rendering: citation markers become `[n](#cite-n)` links (rendered as chips), and
 * the "Not covered by the documents:" sentence is split off so it can be shown on its own. Rewriting markers before
 * parsing, instead of splitting the string around them, keeps the model's paragraphs and lists intact.
 */
export function prepareAnswer(answer: string, contextSize: number): { body: string; notCovered: string | null } {
  const segments = parseAnswer(answer, contextSize);
  const body = toMarkdown(segments.filter((s) => !s.notCovered)).trim();
  const missing = segments.filter((s) => s.notCovered);
  return { body, notCovered: missing.length ? toMarkdown(missing).trim() : null };
}
