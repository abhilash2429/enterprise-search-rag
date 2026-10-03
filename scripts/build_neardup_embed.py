"""Embedding near-duplicates.

  --analyze    cosine of labeled pairs per pooling and split (positives: conflicting_info gold pairs,
               hard negatives: completeness and project_related gold pairs). Pick pooling and threshold on dev.
  --build      kNN pairs for one pooling, cluster stats at several thresholds, sample pairs per band.
"""
import argparse
import itertools
import random

import numpy as np
import pyarrow.parquet as pq

from entsearch.data import CORPUS, SPLITS, load_questions
from entsearch.index.neardup import clusters
from entsearch.index.neardup_embed import OUT, doc_vectors, knn_pairs, pair_cosines

POSITIVE_TYPES = ("conflicting_info",)
NEGATIVE_TYPES = ("completeness", "project_related")
THRESHOLDS = (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98)


def labeled_pairs():
    q = load_questions()
    dev = set((SPLITS / "dev.txt").read_text().split())
    rows = []
    for r in q.itertuples():
        if r.question_type in POSITIVE_TYPES + NEGATIVE_TYPES:
            for a, b in itertools.combinations(sorted(set(r.expected_doc_ids)), 2):
                rows.append((a, b, r.question_type in POSITIVE_TYPES, "dev" if r.question_id in dev else "test"))
    return rows


def analyze() -> None:
    rows = labeled_pairs()
    for pooling in ("mean", "first"):
        ids, vecs = doc_vectors(pooling=pooling)
        cos = pair_cosines(ids, vecs, [(a, b) for a, b, _, _ in rows])
        print(f"\n== pooling={pooling}")
        for split in ("dev", "test"):
            pos = np.array([c for c, (_, _, y, s) in zip(cos, rows) if y and s == split])
            neg = np.array([c for c, (_, _, y, s) in zip(cos, rows) if not y and s == split])
            print(f"{split}: positives n={len(pos)} mean {pos.mean():.3f} min {pos.min():.3f} | "
                  f"negatives n={len(neg)} mean {neg.mean():.3f} max {neg.max():.3f}")
            print("   threshold  " + "  ".join(f"{t:.2f}" for t in THRESHOLDS))
            print("   pos >= t   " + "  ".join(f"{(pos >= t).mean():.2f}" for t in THRESHOLDS))
            print("   neg >= t   " + "  ".join(f"{(neg >= t).mean():.2f}" for t in THRESHOLDS))


def build(pooling: str, floor: float) -> None:
    ids, vecs = doc_vectors(pooling=pooling)
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = knn_pairs(vecs, floor=floor)
    np.save(OUT / f"pairs_{pooling}_floor{floor}.npy", pairs)
    (OUT / "doc_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf8")
    n = len(ids)
    gold = {d for ds in load_questions().expected_doc_ids for d in ds}
    gold_idx = np.array([i for i, d in enumerate(ids) if d in gold])
    meta = pq.read_table(CORPUS, columns=["doc_id", "source_type", "title"]).to_pandas().drop_duplicates("doc_id").set_index("doc_id")

    lines = [f"# Embedding near-duplicate report ({pooling} pooling)\n\n{n:,} docs, {len(pairs):,} kNN pairs (k=10) with cosine >= {floor}\n",
             "| threshold | pairs | docs in clusters | clusters (size>=2) | max size | p99 size | gold docs with a near-dup |",
             "|---|---|---|---|---|---|---|"]
    for t in [x for x in THRESHOLDS if x >= floor]:
        c = clusters(n, pairs, t, field="cos")
        _, inv, counts = np.unique(c, return_inverse=True, return_counts=True)
        size, multi = counts[inv], counts[counts >= 2]
        lines.append(f"| {t} | {(pairs['cos'] >= t).sum():,} | {(size >= 2).sum():,} | {len(multi):,} | "
                     f"{multi.max() if len(multi) else 0:,} | {int(np.percentile(multi, 99)) if len(multi) else 0} | "
                     f"{(size[gold_idx] >= 2).sum()}/{len(gold_idx)} |")
    rng = random.Random(0)
    lines.append("\n## Sample pairs per band\n")
    edges = list(THRESHOLDS) + [1.01]
    for lo, hi in zip(edges, edges[1:]):
        sel = pairs[(pairs["cos"] >= lo) & (pairs["cos"] < hi)]
        lines.append(f"### cosine [{lo}, {min(hi, 1.0)}): {len(sel):,} pairs\n")
        for i, j, s in rng.sample(list(sel), min(5, len(sel))):
            a, b = meta.loc[ids[i]], meta.loc[ids[j]]
            lines.append(f"- **{s:.3f}** `{ids[i]}` ({a.source_type}) {a.title[:90]!r}  \n  vs `{ids[j]}` ({b.source_type}) {b.title[:90]!r}")
        lines.append("")
    (OUT / f"report_{pooling}.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines[:14]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--pooling", choices=["mean", "first"], default="mean")
    p.add_argument("--floor", type=float, default=0.8)
    args = p.parse_args()
    if args.analyze:
        analyze()
    if args.build:
        build(args.pooling, args.floor)


if __name__ == "__main__":
    main()
