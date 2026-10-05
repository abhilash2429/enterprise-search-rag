"""Answer confidence flag as a detector of wrong answers (judge label, seed 0).

flag = verifier verdict not fully supported, OR reranker top score < tau. tau is picked on dev only (max F1 for
"answer is wrong", -inf meaning verifier alone) and saved to runs/<dev run>/flagger.json; the test step reads that
file and never picks anything. Refusals and verifier parse errors are excluded and counted.

  python scripts/eval_flagger.py dev rerank_router_d100
  python scripts/eval_flagger.py test test_router_d100 --tau-from rerank_router_d100
"""
import argparse
import json

import numpy as np
import pandas as pd

from entsearch.answer import read_jsonl
from entsearch.data import ROOT, load_questions
from entsearch.stats import bootstrap_ci

p = argparse.ArgumentParser()
p.add_argument("split", choices=["dev", "test"])
p.add_argument("run")
p.add_argument("--tau-from", help="dev run whose flagger.json holds tau (required on test)")
args = p.parse_args()
if args.split == "test" and not args.tau_from:
    raise SystemExit("test takes tau from dev: pass --tau-from <dev run>")

run_dir = ROOT / "runs" / args.run
q = load_questions(args.split).set_index("question_id")
ev = pd.DataFrame(json.loads((run_dir / "eval_results.json").read_text(encoding="utf8"))["questions"]).set_index("question_id")
ans = pd.DataFrame(read_jsonl(run_dir / "answers.jsonl")).set_index("question_id")
ver = pd.DataFrame(read_jsonl(run_dir / "verify.jsonl")).drop_duplicates("question_id", keep="last").set_index("question_id")

d = pd.DataFrame(index=q.index)
d["type"] = q.question_type
d["wrong"] = ~ev.answer_correct.reindex(d.index).astype(bool)
d["rerank_top"] = ans.rerank_top.reindex(d.index)
d["abstained"] = ans.abstained.reindex(d.index).astype(bool)
d["verified"] = ver.get("verdict").reindex(d.index).notna()
d["verifier_flag"] = ver.get("flagged").reindex(d.index).fillna(False).astype(bool)
d["verdict"] = ver.get("verdict").reindex(d.index)
excluded = {"abstained": int(d.abstained.sum()), "not verified (parse error or missing)": int((~d.abstained & ~d.verified).sum())}
d = d[~d.abstained & d.verified]


def auroc(score: pd.Series, positive: pd.Series) -> float:
    """P(a wrong answer scores lower than a right one), ties half; via ranks (Mann-Whitney)."""
    r = (-score).rank(method="average")
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def stats(flag: pd.Series) -> dict:
    tp, fp = int((flag & d.wrong).sum()), int((flag & ~d.wrong).sum())
    fn = int((~flag & d.wrong).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"flagged": int(flag.sum()), "catch": rec, "precision": prec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "false_alarm": fp / int((~d.wrong).sum()), "correct_if_unflagged": float(1 - d.wrong[~flag].mean()),
            "correct_if_flagged": float(1 - d.wrong[flag].mean()) if flag.any() else float("nan")}


if args.split == "dev":
    taus = [-np.inf] + sorted(d.rerank_top.unique())
    f1 = [stats(d.verifier_flag | (d.rerank_top < t))["f1"] for t in taus]
    tau = float(taus[int(np.argmax(f1))])
    (run_dir / "flagger.json").write_text(json.dumps({"tau": tau, "picked_on": args.run, "criterion": "max F1, wrong answers"}))
else:
    tau = json.loads((ROOT / "runs" / args.tau_from / "flagger.json").read_text())["tau"]

flag = d.verifier_flag | (d.rerank_top < tau)
rows = {
    "verifier only": stats(d.verifier_flag),
    f"rerank top < {tau:.3f} only": stats(d.rerank_top < tau),
    "combined (shipped)": stats(flag),
}
print(f"{args.split} {args.run}: n={len(d)}, wrong={int(d.wrong.sum())} ({d.wrong.mean():.1%}), excluded {excluded}")
print(f"tau = {tau:.4f} ({'picked here, max F1' if args.split == 'dev' else 'from ' + args.tau_from})")
print(f"AUROC of reranker top score for wrong answers: {auroc(d.rerank_top, d.wrong):.3f}")
print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.3f}"))

lo_hi = {}
for name, s in [("catch", flag[d.wrong]), ("false_alarm", flag[~d.wrong]),
                ("correct_if_unflagged", ~d.wrong[~flag]), ("correct_if_flagged", ~d.wrong[flag])]:
    if len(s):
        m, lo, hi = bootstrap_ci(s.astype(float).to_numpy())
        lo_hi[name] = [round(m, 3), round(lo, 3), round(hi, 3)]
print("combined, 95% CI:", lo_hi)
print("verdicts:", d.verdict.value_counts().to_dict())
by_type = d.assign(flag=flag).groupby("type").agg(n=("wrong", "size"), wrong=("wrong", "mean"), flagged=("flag", "mean"))
print(by_type.to_string(float_format=lambda x: f"{x:.2f}"))
(run_dir / "flagger_report.json").write_text(json.dumps(
    {"split": args.split, "tau": tau, "n": len(d), "excluded": excluded, "auroc_rerank_top": auroc(d.rerank_top, d.wrong),
     "rows": rows, "combined_ci": lo_hi}, indent=1))
