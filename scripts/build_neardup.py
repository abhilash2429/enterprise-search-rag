"""MinHash signatures, LSH pairs down to a floor, and a threshold report with sample pairs per similarity band."""
import argparse
import random

import numpy as np
import pyarrow.parquet as pq

from entsearch.data import CORPUS, load_questions
from entsearch.index.neardup import OUT, build_signatures, candidate_pairs, clusters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--floor", type=float, default=0.5)
    args = p.parse_args()

    if not (OUT / "signatures.npy").exists():
        build_signatures()
    sigs = np.load(OUT / "signatures.npy")
    doc_ids = (OUT / "doc_ids.txt").read_text(encoding="utf8").split("\n")[:-1]
    pairs_path = OUT / f"pairs_floor{args.floor}.npy"
    if not pairs_path.exists():
        np.save(pairs_path, candidate_pairs(sigs, args.floor))
    pairs = np.load(pairs_path)
    n = len(doc_ids)

    meta = pq.read_table(CORPUS, columns=["doc_id", "source_type", "title", "content"]).to_pandas().drop_duplicates("doc_id").set_index("doc_id")
    gold = {d for ds in load_questions().expected_doc_ids for d in ds}
    gold_idx = np.array([i for i, d in enumerate(doc_ids) if d in gold])

    lines = [f"# Near-duplicate report\n\n{n:,} docs, {len(pairs):,} pairs with estimated Jaccard >= {args.floor}\n",
             "| threshold | pairs | docs in clusters | clusters (size>=2) | max size | p99 size | gold docs with a near-dup |",
             "|---|---|---|---|---|---|---|"]
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        if t < args.floor:
            continue
        c = clusters(n, pairs, t)
        _, inv, counts = np.unique(c, return_inverse=True, return_counts=True)
        size = counts[inv]
        multi = counts[counts >= 2]
        lines.append(
            f"| {t} | {(pairs['jaccard'] >= t).sum():,} | {(size >= 2).sum():,} | {len(multi):,} | "
            f"{multi.max() if len(multi) else 0:,} | {int(np.percentile(multi, 99)) if len(multi) else 0} | "
            f"{(size[gold_idx] >= 2).sum()}/{len(gold_idx)} |"
        )

    rng = random.Random(0)
    bands = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.01)]
    lines.append("\n## Sample pairs per band\n")
    for lo, hi in bands:
        sel = pairs[(pairs["jaccard"] >= lo) & (pairs["jaccard"] < hi)]
        lines.append(f"### Jaccard [{lo}, {min(hi, 1.0)}{']' if hi > 1 else ')'}: {len(sel):,} pairs\n")
        for i, j, s in rng.sample(list(sel), min(5, len(sel))):
            a, b = meta.loc[doc_ids[i]], meta.loc[doc_ids[j]]
            lines.append(f"- **{s:.2f}** `{doc_ids[i]}` ({a.source_type}) {a.title[:90]!r}  \n  vs `{doc_ids[j]}` ({b.source_type}) {b.title[:90]!r}")
        lines.append("")

    (OUT / "report.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines[:12]))
    print(f"\nfull report with samples: {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
