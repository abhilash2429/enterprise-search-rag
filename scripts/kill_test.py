"""Kill test on dev: each config vs the BM25 baseline, per-question scores averaged over generation seeds.

Per-question score = harness combined score: completeness_pct if the judge marks the answer correct, else 0.
Seed 0 of config X is runs/X, seed k is runs/X_s<k>. Paired bootstrap over dev questions present in every run.
Pass criteria (PROJECT.md section 7): system score >= 55, and the paired CI vs BM25 excludes 0.

  python scripts/kill_test.py bm25_opensearch rerank_qwen3_d100 rerank_qwen3_d100_cited --seeds 3
"""
import argparse
import json

import pandas as pd

from entsearch.answer import read_jsonl
from entsearch.data import ROOT, load_questions
from entsearch.stats import bootstrap_ci, paired_bootstrap

p = argparse.ArgumentParser()
p.add_argument("baseline")
p.add_argument("systems", nargs="*")
p.add_argument("--seeds", type=int, default=3)
p.add_argument("--split", default="dev")
args = p.parse_args()

qids = set(load_questions(args.split).question_id)


def per_question(run: str) -> pd.DataFrame:
    """question_id x {combined, correct}, mean over the seeds that exist; also the seed count and $/question."""
    frames, costs = [], []
    for k in range(args.seeds):
        d = ROOT / "runs" / (run if k == 0 else f"{run}_s{k}")
        if not (d / "eval_results.json").exists():
            continue
        res = pd.DataFrame(json.loads((d / "eval_results.json").read_text(encoding="utf8"))["questions"])
        res = res[res.question_id.isin(qids)]
        res["combined"] = res.completeness_pct.where(res.answer_correct, 0.0)
        res["correct"] = res.answer_correct.astype(float) * 100
        frames.append(res.set_index("question_id")[["combined", "correct"]])
        usage = pd.DataFrame(read_jsonl(d / "usage.jsonl"))
        if "question_id" in usage:
            gen = usage[(usage.stage == "generate") & usage.question_id.isin(qids)]
            costs.append(gen.cost_usd.sum() / max(1, gen.question_id.nunique()))
    if not frames:
        raise SystemExit(f"no eval_results.json for {run}")
    out = sum(frames) / len(frames)
    out.attrs["seeds"], out.attrs["gen_cost"] = len(frames), sum(costs) / len(costs) if costs else float("nan")
    return out.dropna()


base = per_question(args.baseline)
rows = []
for run in [args.baseline, *args.systems]:
    s = per_question(run)
    common = s.index.intersection(base.index)
    mean, lo, hi = bootstrap_ci(s.combined)
    row = {"run": run, "seeds": s.attrs["seeds"], "n": len(s), "combined": mean, "ci": f"[{lo:.1f}, {hi:.1f}]",
           "correct": s.correct.mean(), "gen $/q": s.attrs["gen_cost"]}
    if run != args.baseline:
        d, dlo, dhi = paired_bootstrap(s.combined[common], base.combined[common])
        row |= {"vs base": f"{d:+.1f} [{dlo:+.1f}, {dhi:+.1f}]", "pass": mean >= 55 and dlo > 0}
    rows.append(row)

pd.set_option("display.width", 200)
print(f"{args.split}: combined = completeness if judged correct else 0, mean over seeds per question")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.1f}" if abs(x) >= 0.1 else f"{x:.4f}"))
