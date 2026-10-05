"use client"

import { useEffect, useState, useSyncExternalStore } from "react"

/** performance.now(), refreshed every `ms` while `active`. */
export function useNow(active: boolean, ms = 100): number {
  const [now, setNow] = useState(0)
  useEffect(() => {
    if (!active) return
    const id = setInterval(() => setNow(performance.now()), ms)
    return () => clearInterval(id)
  }, [active, ms])
  return now
}

function subscribeRoot(onChange: () => void): () => void {
  const observer = new MutationObserver(onChange)
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class", "data-record"] })
  return () => observer.disconnect()
}

export type Theme = "light" | "dark"

/** The theme on <html> (set before paint by the layout script) and a setter that stores the choice. */
export function useTheme(): [Theme, (t: Theme) => void] {
  const theme = useSyncExternalStore(
    subscribeRoot,
    () => (document.documentElement.classList.contains("dark") ? "dark" : "light"),
    () => "light" as const
  )
  const set = (t: Theme) => {
    document.documentElement.classList.toggle("dark", t === "dark")
    try {
      localStorage.setItem("theme", t)
    } catch {
      // storage unavailable: the choice lasts for this page only
    }
  }
  return [theme, set]
}

/** True in recording mode (?record=1). */
export function useRecordMode(): boolean {
  return useSyncExternalStore(
    subscribeRoot,
    () => document.documentElement.dataset.record !== undefined,
    () => false
  )
}
