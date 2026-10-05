"use client"

import { X } from "lucide-react"
import { useEffect, useRef } from "react"

import type { Document, Source } from "@/lib/api"

import { SourceTag } from "./source-dot"

export type ViewerState = {
  rank: number
  score: number
  doc_id: string
  title: string
  source: Source
  doc: Document | null
  error: string | null
}

export function DocumentViewer({ state, onClose }: { state: ViewerState | null; onClose: () => void }) {
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const open = state !== null

  useEffect(() => {
    if (!open) return
    const trigger = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = "hidden"
    closeButtonRef.current?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault()
        onClose()
      }
    }
    document.addEventListener("keydown", onKey)
    return () => {
      document.removeEventListener("keydown", onKey)
      document.body.style.overflow = previousOverflow
      trigger?.focus()
    }
  }, [open, onClose])

  if (!state) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/68 p-4 backdrop-blur-[2px]"
      onClick={(event) => {
        if (event.currentTarget === event.target) onClose()
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="document-viewer-title"
        data-role="drawer"
        className="flex max-h-[calc(100dvh-2rem)] w-full max-w-4xl flex-col overflow-hidden rounded-[22px] bg-white shadow-[0_30px_90px_rgba(15,15,15,0.38)] outline-none"
      >
        <header className="flex items-center justify-between gap-4 px-5 py-3.5">
          <div className="min-w-0">
            <h2 id="document-viewer-title" className="truncate text-[15px] font-bold text-slate-950">
              {state.title}
            </h2>
            <p className="mt-0.5 flex items-center gap-2 text-[12px] font-medium text-slate-500">
              <SourceTag source={state.source} />
              <span className="font-mono text-[11px]">{state.doc_id}</span>
            </p>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            aria-label="Close document"
            className="inline-flex size-9 shrink-0 items-center justify-center rounded-full bg-neutral-100 text-neutral-600 transition-colors hover:bg-neutral-200 hover:text-neutral-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-500/50"
          >
            <X className="size-4.5" aria-hidden />
          </button>
        </header>

        <div className="min-h-0 flex-1 overflow-auto bg-neutral-50 px-6 py-5">
          {state.error ? (
            <p className="text-[14px] text-neutral-700">Could not load the document: {state.error}</p>
          ) : state.doc ? (
            <pre
              className="whitespace-pre-wrap break-words font-sans text-[14px] leading-relaxed text-neutral-800"
              data-role="document-content"
            >
              {state.doc.content}
            </pre>
          ) : (
            <div className="flex animate-pulse flex-col gap-2.5">
              {[100, 92, 97, 80, 95, 60].map((w, i) => (
                <div key={i} className="h-3.5 rounded bg-neutral-200" style={{ width: `${w}%` }} />
              ))}
            </div>
          )}
        </div>

        <footer className="flex flex-wrap items-center gap-x-3 gap-y-1 bg-neutral-100 px-5 py-3 text-[12px]">
          <span className="rounded bg-neutral-800 px-1.5 py-0.5 font-semibold text-white">{state.rank}</span>
          <span className="font-bold uppercase tracking-[0.08em] text-slate-700">Rank {state.rank} of 10</span>
          <span className="text-slate-300" aria-hidden>
            /
          </span>
          <span className="font-medium tabular-nums text-slate-600">relevance {state.score.toFixed(3)}</span>
          {state.doc && (
            <>
              <span className="text-slate-300" aria-hidden>
                /
              </span>
              <span className="font-medium tabular-nums text-slate-600">
                {state.doc.content.length.toLocaleString()} characters
              </span>
            </>
          )}
        </footer>
      </div>
    </div>
  )
}
