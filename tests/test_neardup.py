import random

import numpy as np

from entsearch.index.neardup import candidate_pairs, clusters, signature


def _doc(rng, n=300):
    words = [f"w{i}" for i in range(2000)]
    return " ".join(rng.choice(words) for _ in range(n))


def test_near_copies_cluster_and_unrelated_do_not():
    rng = random.Random(0)
    a = _doc(rng)
    a_edit = a.replace(a.split()[150], "changedfact", 1)  # one fact differs
    b = _doc(rng)
    sigs = np.stack([signature(t) for t in (a, a_edit, b, a)])
    pairs = candidate_pairs(sigs, floor=0.5)
    got = {(int(i), int(j)): float(s) for i, j, s in pairs}
    assert got[(0, 3)] == 1.0
    assert got[(0, 1)] > 0.9
    assert not any(2 in k for k in got)
    c = clusters(4, pairs, threshold=0.9)
    assert c[0] == c[1] == c[3] != c[2]
    assert len(set(clusters(4, pairs, threshold=1.0))) == 3
