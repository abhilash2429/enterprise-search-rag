"use client"

import { useMemo } from "react"
import type { Components } from "react-markdown"

import { Markdown } from "@/components/ui/markdown"
import { prepareAnswer } from "@/lib/citations"
import { cn } from "@/lib/utils"

const PROSE =
  "min-w-0 max-w-none break-words text-[16px] leading-[1.65] [&_h1]:mb-2 [&_h1]:mt-4 [&_h1]:text-lg [&_h1]:font-semibold [&_h2]:mb-2 [&_h2]:mt-4 [&_h2]:font-semibold [&_li]:my-1 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_p]:my-2 [&_strong]:font-semibold [&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_code]:rounded [&_code]:bg-neutral-100 [&_code]:px-1 [&_code]:text-[0.9em]"

export function CitationChip({
  n,
  active,
  onCite,
}: {
  n: number
  active?: boolean
  onCite: (n: number) => void
}) {
  return (
    <button
      type="button"
      data-cite={n}
      onClick={() => onCite(n)}
      aria-label={`Show document ${n}`}
      className={cn(
        "mx-0.5 inline-flex min-w-[1.35rem] cursor-pointer items-center justify-center rounded px-1.5 align-[1px] text-[11px] font-semibold tabular-nums no-underline transition-colors",
        active ? "bg-neutral-800 text-white" : "bg-neutral-100 text-neutral-700 hover:bg-neutral-200 hover:text-neutral-950"
      )}
    >
      {n}
    </button>
  )
}

export function CitationChips({ ns, active, onCite }: { ns: number[]; active: number | null; onCite: (n: number) => void }) {
  return (
    <span className="whitespace-nowrap">
      {ns.map((n) => (
        <CitationChip key={n} n={n} active={active === n} onCite={onCite} />
      ))}
    </span>
  )
}

function useComponents(active: number | null, onCite: (n: number) => void) {
  return useMemo<Partial<Components>>(
    () => ({
      a({ href, children, ...props }) {
        const target = typeof href === "string" ? href : ""
        if (!target.startsWith("#cite-")) {
          return (
            <a
              href={target}
              target="_blank"
              rel="noreferrer"
              className="text-neutral-800 underline decoration-neutral-400 underline-offset-2"
              {...props}
            >
              {children}
            </a>
          )
        }
        const n = Number(target.slice("#cite-".length))
        return <CitationChip n={n} active={active === n} onCite={onCite} />
      },
      table({ children }) {
        return (
          <div className="my-3 max-w-full overflow-x-auto rounded-lg bg-neutral-50">
            <table className="w-full border-collapse text-left text-[14px] [&_td]:px-3 [&_td]:py-2 [&_th]:bg-neutral-100 [&_th]:px-3 [&_th]:py-2 [&_th]:font-semibold [&_tr:nth-child(even)]:bg-neutral-100/60">
              {children}
            </table>
          </div>
        )
      },
    }),
    [active, onCite]
  )
}

/**
 * The answer as one markdown document with citation chips. The "Not covered by the documents:" sentence is shown
 * apart, so a partial answer says plainly what it could not answer.
 */
export function AnswerBody({
  answer,
  contextSize,
  active,
  onCite,
}: {
  answer: string
  contextSize: number
  active: number | null
  onCite: (n: number) => void
}) {
  const { body, notCovered } = useMemo(() => prepareAnswer(answer, contextSize), [answer, contextSize])
  const components = useComponents(active, onCite)

  return (
    <div className="flex flex-col gap-3">
      <div className={PROSE} data-role="answer">
        <Markdown components={components}>{body}</Markdown>
      </div>
      {notCovered && (
        <div
          className={cn(PROSE, "rounded-xl bg-neutral-100 px-3.5 py-1 text-[14px] text-neutral-700")}
          data-role="not-covered"
        >
          <Markdown components={components}>{notCovered}</Markdown>
        </div>
      )}
    </div>
  )
}
