import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { parseAskEvent, type Generate } from "@/lib/api";
import { citedOrder, parseAnswer, prepareAnswer, type AnswerSegment } from "@/lib/citations";
import { FIXTURES_DIR } from "@/lib/mock";

const cites = (s: AnswerSegment[]) => s.filter((x) => x.kind === "cite").map((x) => (x.kind === "cite" ? x.ns : []));
const text = (s: AnswerSegment[]) => s.map((x) => (x.kind === "text" ? x.text : "|")).join("");

describe("parseAnswer", () => {
  it.each([
    ["A [1].", [[1]], "A |."],
    ["A [1][2].", [[1, 2]], "A |."],
    ["A [1, 2].", [[1, 2]], "A |."],
    ["A [3,4] and [5].", [[3, 4], [5]], "A | and |."],
    ["A 【2】.", [[2]], "A |."],
    ["A 【1†source】.", [[1]], "A |."],
    ["A [1; 2].", [[1, 2]], "A |."],
    ["A [1][1].", [[1]], "A |."],
    ["A [2] [2].", [[2], [2]], "A | |."],
  ])("%s", (answer, groups, shape) => {
    const s = parseAnswer(answer, 10);
    expect(cites(s)).toEqual(groups);
    expect(text(s)).toBe(shape);
  });

  it("leaves numbers outside the context as text", () => {
    expect(cites(parseAnswer("A [11] and [0].", 10))).toEqual([]);
    expect(text(parseAnswer("A [11] and [0].", 10))).toBe("A [11] and [0].");
    expect(cites(parseAnswer("A [2, 11].", 10))).toEqual([[2]]);
  });

  it("keeps newlines", () => {
    expect(text(parseAnswer("One [1].\n\n* Two [2].", 10))).toBe("One |.\n\n* Two |.");
  });

  it("marks the not-covered sentence to the end of its line", () => {
    const s = parseAnswer("Yes [1]. Not covered by the documents: the date [2].\nNext line.", 10);
    expect(s.filter((x) => x.notCovered).map((x) => (x.kind === "text" ? x.text : "|")).join("")).toBe(
      "Not covered by the documents: the date |.",
    );
    expect(s.at(-1)).toMatchObject({ kind: "text", text: "\nNext line.", notCovered: false });
  });
});

describe("every recorded answer", () => {
  const runs = readdirSync(path.join(FIXTURES_DIR, "ask"));
  it.each(runs)("%s: chips match generate.citations in first-cited order", (file) => {
    const line = readFileSync(path.join(FIXTURES_DIR, "ask", file), "utf8")
      .split(/\r?\n/)
      .map((l) => (l.trim() ? JSON.parse(l) : null))
      .find((e) => e?.event === "generate");
    const g = parseAskEvent("generate", line.data).data as Generate;
    const s = parseAnswer(g.answer, g.context.length);
    expect(citedOrder(s)).toEqual(g.citations.map((c) => c.n));
    // No marker survives as text, and the text round-trips apart from the markers.
    expect(text(s)).not.toMatch(/[\[【]\d/);
    expect(s.some((x) => x.notCovered)).toBe(g.partial);
  });
});

describe("prepareAnswer", () => {
  it("links markers and splits off the not-covered sentence", () => {
    expect(prepareAnswer("Yes **bold** [1][2].\n\nNot covered by the documents: the date [3].", 10)).toEqual({
      body: "Yes **bold** [1](#cite-1)[2](#cite-2).",
      notCovered: "Not covered by the documents: the date [3](#cite-3).",
    });
    expect(prepareAnswer("Due 17:00 UTC【1】. Not covered by the documents: nothing else.", 10)).toEqual({
      body: "Due 17:00 UTC[1](#cite-1).",
      notCovered: "Not covered by the documents: nothing else.",
    });
    expect(prepareAnswer("No refs.", 10)).toEqual({ body: "No refs.", notCovered: null });
  });
});
