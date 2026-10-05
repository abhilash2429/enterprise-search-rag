"use client"

import { useEffect, useState } from "react"

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
