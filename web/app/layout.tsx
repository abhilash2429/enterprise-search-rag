import type { Metadata } from "next"
import { GeistMono } from "geist/font/mono"
import { GeistSans } from "geist/font/sans"

import { TooltipProvider } from "@/components/ui/tooltip"
import "./globals.css"

export const metadata: Metadata = {
  title: "Enterprise Search - cited answers over company documents",
  description: "Hybrid retrieval, reranking, cited answers and a confidence check over 512K company documents.",
}

// Runs before first paint: theme from ?theme= or the stored choice, ?record=1 recording mode, and in recording mode a
// cursor that hides after 2 s without movement.
const INIT_SCRIPT = `(function(){var d=document.documentElement,q=new URLSearchParams(location.search),t=q.get("theme");
try{if(t!=="light"&&t!=="dark")t=localStorage.getItem("theme")}catch(e){}
if(t==="dark")d.classList.add("dark");
if(q.get("record")==="1"){d.dataset.record="";var h;var wake=function(){delete d.dataset.idle;clearTimeout(h);h=setTimeout(function(){d.dataset.idle=""},2000)};
addEventListener("mousemove",wake,{passive:true});addEventListener("mousedown",wake,{passive:true});wake()}})()`

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable} h-full antialiased`} suppressHydrationWarning>
      <body className="flex min-h-full flex-col">
        <script dangerouslySetInnerHTML={{ __html: INIT_SCRIPT }} />
        <TooltipProvider>{children}</TooltipProvider>
      </body>
    </html>
  )
}
