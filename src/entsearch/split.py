import numpy as np
import pandas as pd

SEED = 20261003
DEV_SIZE = 150


def stratified_split(q: pd.DataFrame, dev_size: int = DEV_SIZE, seed: int = SEED) -> tuple[list[str], list[str]]:
    """Dev/test question_id lists, stratified by question_type with largest-remainder allocation."""
    counts = q.question_type.value_counts().sort_index()
    exact = counts * dev_size / len(q)
    alloc = np.floor(exact).astype(int)
    remainder = (exact - alloc).sort_values(ascending=False, kind="stable")
    alloc[remainder.index[: dev_size - alloc.sum()]] += 1

    rng = np.random.default_rng(seed)
    dev, test = [], []
    for qtype, n_dev in alloc.items():
        ids = sorted(q.loc[q.question_type == qtype, "question_id"])
        perm = rng.permutation(len(ids))
        dev += [ids[i] for i in perm[:n_dev]]
        test += [ids[i] for i in perm[n_dev:]]
    return sorted(dev), sorted(test)
