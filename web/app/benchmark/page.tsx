import type { Metadata } from "next"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

import { RESULT_GROUPS, type ResultTable } from "@/lib/results"
import { cn } from "@/lib/utils"

export const metadata: Metadata = {
  title: "Benchmark results - Enterprise Search",
  description: "Held-out test, dev ablations, serving latency and the confidence flag, from the repository README.",
}

/** README captions use `code` spans; render them as code, everything else as text. */
function Caption({ text }: { text: string }) {
  return (
    <p className="text-[13px] leading-relaxed text-neutral-600">
      {text.split(/(`[^`]+`)/).map((part, i) =>
        part.startsWith("`") ? (
          <code key={i} className="rounded bg-neutral-100 px-1 font-mono text-[12px] text-neutral-800">
            {part.slice(1, -1)}
          </code>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </p>
  )
}

function Table({ table }: { table: ResultTable<string> }) {
  return (
    <section className="flex flex-col gap-2.5" aria-labelledby={`t-${table.id}`} data-table={table.id}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 id={`t-${table.id}`} className="text-[15px] font-semibold tracking-[-0.01em] text-neutral-900">
          {table.title}
        </h3>
        <span className="text-[10.5px] font-semibold uppercase tracking-[0.09em] text-neutral-600">
          README: {table.section}
        </span>
      </div>
      <Caption text={table.caption} />
      <div className="max-w-full overflow-x-auto rounded-xl bg-neutral-50">
        <table className="w-full border-collapse text-left text-[14px]">
          <thead>
            <tr>
              {table.columns.map((c, i) => (
                <th
                  key={c.key}
                  scope="col"
                  className={cn("bg-neutral-100 px-3 py-2 font-semibold text-neutral-800", i > 0 && "whitespace-nowrap")}
                >
                  {c.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, r) => (
              <tr
                key={r}
                data-headline={r === table.headline || undefined}
                className={cn(
                  "border-t border-neutral-200/70",
                  r === table.headline ? "bg-white font-semibold text-neutral-900" : "text-neutral-700"
                )}
              >
                {table.columns.map((c, i) => (
                  <td key={c.key} className={cn("px-3 py-2 align-top", i > 0 && "tabular-nums")}>
                    {i === 0 && r === table.headline && (
                      <span className="mr-2 rounded-full bg-neutral-800 px-2 py-0.5 align-[1px] text-[10px] font-bold uppercase tracking-[0.06em] text-white">
                        Headline
                      </span>
                    )}
                    <span data-cell>{row[c.key]}</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

export default function BenchmarkPage() {
  return (
    <div className="app-canvas min-h-dvh p-3 text-neutral-700 lg:p-4">
      <main className="mx-auto flex max-w-[1240px] flex-col gap-10 rounded-[24px] bg-white px-8 py-8 lg:px-12">
        <header className="flex flex-col gap-3">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 self-start rounded text-[12px] font-medium text-neutral-600 hover:text-neutral-900"
          >
            <ArrowLeft className="size-4" aria-hidden />
            Back to search
          </Link>
          <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-neutral-600">
            EnterpriseRAG-Bench · 511,958 documents · 500 questions
          </p>
          <h1 className="text-[34px] font-semibold leading-tight tracking-[-0.04em] text-neutral-900">Benchmark results</h1>
          <p className="max-w-[760px] text-[14px] leading-relaxed text-neutral-600">
            The tables from the repository README, exactly as reported there. Every design choice was ablated on a
            150-question dev split; the final config ran once on the 350-question held-out test split.
          </p>
        </header>
        {RESULT_GROUPS.map((g) => (
          <section key={g.title} className="flex flex-col gap-6" aria-label={g.title}>
            <div className="flex flex-col gap-1 border-b border-neutral-200 pb-2">
              <h2 className="text-[22px] font-semibold tracking-[-0.03em] text-neutral-900">{g.title}</h2>
              {g.note && <p className="text-[13px] text-neutral-600">{g.note}</p>}
            </div>
            {g.tables.map((t) => (
              <Table key={t.id} table={t} />
            ))}
          </section>
        ))}
      </main>
    </div>
  )
}
