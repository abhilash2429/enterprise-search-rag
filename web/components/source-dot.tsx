import type { Source } from "@/lib/api"
import { SOURCE_COLOR, SOURCE_LABEL } from "@/lib/format"
import { cn } from "@/lib/utils"

/** A source name with its colour dot; each of the 9 sources has its own colour. */
export function SourceTag({ source, className }: { source: Source; className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5", className)} data-source={source}>
      <span className="size-2 shrink-0 rounded-full" style={{ backgroundColor: SOURCE_COLOR[source] }} aria-hidden />
      {SOURCE_LABEL[source]}
    </span>
  )
}
