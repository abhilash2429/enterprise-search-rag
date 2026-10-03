"""Judge vs hand labels on correctness: TPR (judge says correct when I said correct) and TNR (judge says incorrect when I said incorrect)."""
import argparse
import json
from pathlib import Path

import pandas as pd

from entsearch.answer import read_jsonl
from entsearch.data import ROOT

p = argparse.ArgumentParser()
p.add_argument("--run", default="bm25_opensearch")
p.add_argument("--labels", default=str(ROOT / "data/labels/judge_validation.jsonl"))
args = p.parse_args()

labels = pd.DataFrame([r for r in read_jsonl(Path(args.labels)) if r["run"] == args.run])
judged = pd.DataFrame(json.loads((ROOT / "runs" / args.run / "eval_results.json").read_text(encoding="utf8"))["questions"])
df = labels.merge(judged[["question_id", "answer_correct", "correctness_reasoning"]], on="question_id", how="left")
assert df.answer_correct.notna().all(), "labeled questions missing from judge results"

human = df.label == "correct"
judge = df.answer_correct.astype(bool)
tp, fn = int((human & judge).sum()), int((human & ~judge).sum())
tn, fp = int((~human & ~judge).sum()), int((~human & judge).sum())
print(f"run {args.run}: {len(df)} labeled ({human.sum()} correct, {(~human).sum()} incorrect by hand)")
print(f"TPR {tp / max(tp + fn, 1):.3f} ({tp}/{tp + fn})  TNR {tn / max(tn + fp, 1):.3f} ({tn}/{tn + fp})  agreement {(tp + tn) / len(df):.3f}")
print(f"judge accuracy-rate {judge.mean():.3f} vs hand {human.mean():.3f}")
dis = df[human != judge]
if len(dis):
    print("\ndisagreements:")
    for r in dis.itertuples():
        print(f"  {r.question_id} [{r.question_type}] hand={r.label} judge={'correct' if r.answer_correct else 'incorrect'} | {r.correctness_reasoning[:150]}")
