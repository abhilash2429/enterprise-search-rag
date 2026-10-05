// lib/results.ts must match README.md cell for cell, under the section each table names.

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { ALL_TABLES } from "@/lib/results";

const README = readFileSync(path.join(process.cwd(), "..", "README.md"), "utf8").split(/\r?\n/);
const cells = (line: string) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

/** Line index where the section path starts and ends: "## A > ### B" or "## A > **B.** paragraph". */
function sectionRange(section: string): [number, number] {
  const parts = section.split(" > ");
  let start = README.findIndex((l) => l === `## ${parts[0]}`);
  expect(start, `## ${parts[0]}`).toBeGreaterThanOrEqual(0);
  let level = 2;
  for (const part of parts.slice(1)) {
    const heading = README.findIndex((l, i) => i > start && l === `### ${part}`);
    const bold = README.findIndex((l, i) => i > start && l.startsWith(`**${part}.**`));
    const at = heading !== -1 ? heading : bold;
    expect(at, part).toBeGreaterThan(start);
    if (heading !== -1) level = 3;
    start = at;
  }
  const end = README.findIndex((l, i) => i > start && /^#{2,3} /.test(l) && l.match(/^#+/)![0].length <= level);
  return [start, end === -1 ? README.length : end];
}

describe.each(ALL_TABLES.map((t) => [t.id, t] as const))("%s", (_id, table) => {
  it("matches the README table under its section exactly", () => {
    const [start, end] = sectionRange(table.section);
    const headers = table.columns.map((c) => c.header);
    const at = README.findIndex((l, i) => i > start && i < end && l.startsWith("|") && cells(l).join("\u0000") === headers.join("\u0000"));
    expect(at, `header row of ${table.id} inside "${table.section}"`).toBeGreaterThan(start);
    expect(README[at + 1]).toMatch(/^\|(---\|)+$/);
    const rows: string[][] = [];
    for (let i = at + 2; README[i]?.startsWith("|"); i++) rows.push(cells(README[i]));
    expect(table.rows.map((r) => table.columns.map((c) => (r as Record<string, string>)[c.key]))).toEqual(rows);
  });

  it("has a caption taken from the README", () => {
    const [start, end] = sectionRange(table.section);
    const prose = README.slice(start, end).join(" ").replace(/\s+/g, " ");
    expect(prose).toContain(table.caption);
  });
});
