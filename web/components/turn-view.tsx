"use client"

import { AnswerBody, CitationChip } from "@/components/answer-body"
import { ConfidenceNote } from "@/components/confidence-note"
import { GoldAnswer } from "@/components/gold-answer"
import { RunFailure } from "@/components/run-failure"
import { StageTrace } from "@/components/stage-trace"
import { WhyPanel } from "@/components/why-panel"
import { Message, MessageContent } from "@/components/ui/message"
import type { Health, Question } from "@/lib/api"
import type { RunState } from "@/lib/run"
import { cn } from "@/lib/utils"

function Badge({ children, dark, role }: { children: React.ReactNode; dark?: boolean; role?: string }) {
  return (
    <span
      data-banner={role}
      className={cn(
        "inline-flex animate-[pop_0.18s_ease-out_both] items-center rounded-full px-2 py-0.5 text-[11px] font-bold uppercase tracking-[0.055em] motion-reduce:animate-none",
        dark ? "bg-neutral-800 text-white" : "bg-neutral-100 text-neutral-600"
      )}
    >
      {children}
    </span>
  )
}

function Skeleton() {
  return (
    <div className="flex animate-pulse flex-col gap-2.5 py-1" data-role="answer-skeleton">
      {[96, 88, 92, 60].map((w, i) => (
        <div key={i} className="h-3.5 rounded bg-neutral-100" style={{ width: `${w}%` }} />
      ))}
    </div>
  )
}

export function TurnView({
  run,
  question,
  health,
  active,
  onCite,
  onShowSources,
  onRetry,
  stepsOpen,
  onToggleSteps,
}: {
  run: RunState
  /** The /questions entry this turn asked, when it was one of them. */
  question: Question | null
  health: Health | null
  active: number | null
  onCite: (turnId: number, n: number) => void
  onShowSources: (turnId: number) => void
  onRetry: (turnId: number) => void
  /** Whether the pipeline steps are expanded; one setting for every turn, so it holds across questions. */
  stepsOpen: boolean
  onToggleSteps: () => void
}) {
  const g = run.events.generate
  const failure = run.outcome && run.outcome.kind !== "done" ? run.outcome : null
  const titles = Object.fromEntries((g?.context ?? []).map((c) => [c.n, c.title]))
  const streaming = run.phase === "checking" || run.phase === "streaming"
  const cite = (n: number) => onCite(run.id, n)

  return (
    <div className="flex flex-col gap-4 py-2" data-turn={run.id}>
      <Message className="justify-end">
        <MessageContent className="max-w-[85%] rounded-[18px] bg-neutral-100 px-4 py-2.5 text-[15px] text-neutral-700">
          {run.question}
        </MessageContent>
      </Message>

      <article className="grid grid-cols-[26px_minmax(0,1fr)] gap-3">
        <div className="flex size-6 items-center justify-center rounded-md bg-neutral-800 text-[10px] font-semibold text-white">
          E
        </div>
        <div className="flex min-w-0 flex-col gap-3 pt-0.5">
          <StageTrace run={run} open={stepsOpen} onToggle={onToggleSteps} />

          {g && (
            <div className="flex flex-wrap gap-1.5">
              {g.abstained ? (
                <Badge dark role="abstained">No answer in the documents</Badge>
              ) : g.partial ? (
                <Badge role="partial">Partial answer</Badge>
              ) : (
                <Badge>Sources cited</Badge>
              )}
            </div>
          )}

          {failure && <RunFailure outcome={failure} health={health} onRetry={() => onRetry(run.id)} />}

          {!g && streaming && <Skeleton />}

          {g && (
            <>
              {g.abstained && (
                <p className="rounded-lg bg-neutral-200 px-3 py-2 text-[13px] font-medium text-neutral-800">
                  None of the 10 retrieved documents answers this, so the system declined instead of guessing.
                </p>
              )}
              <AnswerBody answer={g.answer} titles={titles} active={active} onCite={cite} />
              {g.version_pairs.length > 0 && (
                <ul className="flex flex-col gap-1" data-role="version-pairs">
                  {g.version_pairs.map((p) => (
                    <li key={`${p.a}-${p.b}`} className="text-[12.5px] leading-relaxed text-neutral-600" data-role="version-pair">
                      Documents <CitationChip n={p.a} active={active === p.a} onCite={cite} label={titles[p.a]} /> and{" "}
                      <CitationChip n={p.b} active={active === p.b} onCite={cite} label={titles[p.b]} /> look like versions of the same
                      document; the newer value is used.
                    </li>
                  ))}
                </ul>
              )}
              <ConfidenceNote run={run} active={active} onCite={cite} titles={titles} />
            </>
          )}

          {g && !streaming && (
            <div className="flex flex-wrap items-start gap-x-4 gap-y-2">
              <button
                type="button"
                onClick={() => onShowSources(run.id)}
                className="text-[12px] font-medium text-neutral-600 underline-offset-4 hover:underline"
              >
                View {run.events.rerank?.hits.length ?? 0} sources
              </button>
              <WhyPanel run={run} />
            </div>
          )}

          {question && run.events.fuse && !streaming && (
            <GoldAnswer question={question} run={run} active={active} onCite={cite} />
          )}
        </div>
      </article>
    </div>
  )
}
