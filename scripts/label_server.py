"""Blind hand-labeling of answer correctness for judge validation (TPR/TNR).

Samples N dev questions (stratified by type, fixed seed) from a run, shows question, gold answer and the
candidate answer under the harness's own correctness rubric, and never shows the judge's verdict.
Labels upsert into data/labels/judge_validation.jsonl keyed by (run, question_id); answer_sha guards
against relabeling a changed answer.

    uv run python scripts/label_server.py --run bm25_opensearch
"""

import argparse
import hashlib
import json
import threading
import webbrowser
from pathlib import Path
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from entsearch.answer import read_jsonl
from entsearch.data import ROOT, load_questions
from entsearch.harness import harness_import
from entsearch.split import stratified_split

LABELS = ROOT / "data/labels/judge_validation.jsonl"
SAMPLE_SEED = 7

p = argparse.ArgumentParser()
p.add_argument("--run", default="bm25_opensearch")
p.add_argument("--n", type=int, default=100)
p.add_argument("--port", type=int, default=8765)
p.add_argument("--labels", default=str(LABELS))
p.add_argument("--no-browser", action="store_true")
args = p.parse_args()
LABELS = Path(args.labels)

prompt = harness_import("src.prompts.answer_evaluation").ANSWER_WHOLISTIC_EVALUATION_PROMPT
RUBRIC = prompt.split("Use the following metrics for evaluating the answer:")[1].split("## Query")[0].strip()

answers = {r["question_id"]: r["answer"] for r in read_jsonl(ROOT / "runs" / args.run / "answers.jsonl")}
dev = load_questions("dev")
dev = dev[dev.question_id.isin(answers)]
sample_ids, _ = stratified_split(dev, dev_size=args.n, seed=SAMPLE_SEED)
dev = dev.set_index("question_id")
ITEMS = [
    {
        "question_id": qid,
        "question_type": dev.loc[qid, "question_type"],
        "question": dev.loc[qid, "question"],
        "gold_answer": dev.loc[qid, "gold_answer"],
        "answer": answers[qid],
        "answer_sha": hashlib.sha256(answers[qid].encode()).hexdigest()[:16],
    }
    for qid in sample_ids
]
lock = threading.Lock()


def load_labels() -> dict:
    return {(r["run"], r["question_id"]): r for r in read_jsonl(LABELS)}


def save_label(rec: dict) -> None:
    with lock:
        labels = load_labels()
        labels[(rec["run"], rec["question_id"])] = rec
        LABELS.parent.mkdir(parents=True, exist_ok=True)
        LABELS.write_text("".join(json.dumps(r) + "\n" for r in labels.values()), encoding="utf8")


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Judge Validation Labels</title><style>
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--ok:#15803d;--bad:#b91c1c;--accent:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--muted:#a8a29e;--card:#292524;--line:#44403c;--ok:#4ade80;--bad:#f87171;--accent:#93c5fd}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
main{max-width:920px;margin:0 auto;padding:16px}header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:10px 0}
h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 6px}
.text{white-space:pre-wrap;word-wrap:break-word}button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.ok{border-color:var(--ok);color:var(--ok)}button.bad{border-color:var(--bad);color:var(--bad)}button.sel{color:var(--bg)}button.ok.sel{background:var(--ok)}button.bad.sel{background:var(--bad)}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}input{font:inherit;flex:1;min-width:200px;padding:8px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
details summary{cursor:pointer;color:var(--accent)}.bar{height:6px;background:var(--line);border-radius:3px;flex:1;min-width:120px}.bar>div{height:100%;background:var(--accent);border-radius:3px}
</style></head><body><main>
<header><strong>Judge validation</strong><span class="muted" id="pos"></span><div class="bar"><div id="prog"></div></div><span class="muted" id="count"></span></header>
<details class="card"><summary>Rubric (the judge's own criteria) and keys</summary><div class="text" id="rubric"></div>
<p class="muted">Keys: 1 = correct, 2 = incorrect, j / k = next / previous, u = next unlabeled. Judge verdicts are hidden on purpose.</p></details>
<div class="card"><h2 id="qtype"></h2><div class="text" id="question"></div></div>
<div class="card"><h2>Gold answer</h2><div class="text" id="gold"></div></div>
<div class="card"><h2>Candidate answer</h2><div class="text" id="answer"></div></div>
<div class="card row"><button class="ok" id="b-ok">1 Correct</button><button class="bad" id="b-bad">2 Incorrect</button>
<input id="note" placeholder="Optional note (why)"><button id="prev">k Prev</button><button id="next">j Next</button><button id="unl">u Next unlabeled</button></div>
<p class="muted" id="status"></p></main><script>
let items=[],labels={},i=0,run="";
const $=id=>document.getElementById(id);
async function init(){const d=await (await fetch('/api/state')).json();items=d.items;labels=d.labels;run=d.run;$('rubric').textContent=d.rubric;
 i=Math.max(0,items.findIndex(x=>!labels[x.question_id]));if(i<0)i=0;show()}
function show(){const it=items[i],l=labels[it.question_id];$('pos').textContent=`${i+1} / ${items.length}  ${it.question_id}`;
 $('qtype').textContent=it.question_type.replaceAll('_',' ');$('question').textContent=it.question;$('gold').textContent=it.gold_answer;$('answer').textContent=it.answer;
 $('note').value=l?l.note||'':'';$('b-ok').classList.toggle('sel',!!l&&l.label==='correct');$('b-bad').classList.toggle('sel',!!l&&l.label==='incorrect');
 const n=items.filter(x=>labels[x.question_id]).length;$('count').textContent=`${n} labeled`;$('prog').style.width=(100*n/items.length)+'%';
 $('status').textContent=l&&l.answer_sha!==it.answer_sha?'Answer changed since this label was saved; relabel.':''}
async function label(v){const it=items[i];const rec={run,question_id:it.question_id,question_type:it.question_type,answer_sha:it.answer_sha,label:v,note:$('note').value};
 const r=await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)});
 if(!r.ok){$('status').textContent='Save failed: '+r.status;return}labels[it.question_id]=await r.json();if(i<items.length-1)i++;show()}
function go(d){i=Math.min(items.length-1,Math.max(0,i+d));show()}
function nextUnlabeled(){const j=items.findIndex((x,k)=>k>i&&!labels[x.question_id]);const k=j>=0?j:items.findIndex(x=>!labels[x.question_id]);if(k>=0){i=k;show()}else $('status').textContent='All labeled.'}
$('b-ok').onclick=()=>label('correct');$('b-bad').onclick=()=>label('incorrect');$('prev').onclick=()=>go(-1);$('next').onclick=()=>go(1);$('unl').onclick=nextUnlabeled;
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;({'1':()=>label('correct'),'2':()=>label('incorrect'),'j':()=>go(1),'k':()=>go(-1),'u':nextUnlabeled}[e.key]||(()=>{}))()});
init();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            labels = {qid: r for (run, qid), r in load_labels().items() if run == args.run}
            body = {"run": args.run, "rubric": RUBRIC, "items": ITEMS, "labels": labels}
            self._send(200, json.dumps(body).encode(), "application/json")
        else:
            self._send(404, b"", "text/plain")

    def do_POST(self):
        if self.path != "/api/label":
            return self._send(404, b"", "text/plain")
        rec = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        known = {it["question_id"]: it for it in ITEMS}
        if rec.get("run") != args.run or rec.get("question_id") not in known or rec.get("label") not in ("correct", "incorrect"):
            return self._send(400, b"bad label", "text/plain")
        item = known[rec["question_id"]]
        rec = {"run": args.run, "question_id": item["question_id"], "question_type": item["question_type"],
               "answer_sha": item["answer_sha"], "label": rec["label"], "note": str(rec.get("note", ""))[:500]}
        rec["labeled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_label(rec)
        self._send(200, json.dumps(rec).encode(), "application/json")

    def log_message(self, *a):
        pass


url = f"http://127.0.0.1:{args.port}/"
print(f"{len(ITEMS)} items from run '{args.run}' -> {LABELS}\n{url}")
if not args.no_browser:
    webbrowser.open(url)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
