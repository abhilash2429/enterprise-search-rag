"""Streaming build of a docs x terms raw term-frequency matrix over the whole corpus.

Output under data/index/sparse/:
  tf.npz       scipy CSC matrix, shape (n_docs, n_terms), int32 raw counts
  doc_len.npy  int32 token count per doc (exact, unlike Lucene's lossy norms)
  doc_ids.txt  row order
  vocab.txt    column order, one term per line
"""

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from entsearch.analysis import tokenize
from entsearch.data import ROOT, doc_text, iter_corpus

INDEX_DIR = ROOT / "data/index/sparse"


def _count_batch(texts: list[str]):
    """Per batch: local vocab plus (local_term_id, tf) arrays per doc, to keep IPC small."""
    local: dict[str, int] = {}
    doc_ptr, term_ids, tfs, lengths = [0], [], [], []
    for text in texts:
        toks = tokenize(text)
        lengths.append(len(toks))
        for term, c in Counter(toks).items():
            term_ids.append(local.setdefault(term, len(local)))
            tfs.append(c)
        doc_ptr.append(len(term_ids))
    return (
        list(local),
        np.asarray(doc_ptr, dtype=np.int64),
        np.asarray(term_ids, dtype=np.int32),
        np.asarray(tfs, dtype=np.int32),
        np.asarray(lengths, dtype=np.int32),
    )


def _bounded_map(ex, fn, items, window):
    """Ordered map that keeps at most `window` tasks in flight (Executor.map drains its input up front)."""
    pending = deque()
    for item in items:
        pending.append(ex.submit(fn, item))
        if len(pending) >= window:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def build(out_dir: Path = INDEX_DIR, workers: int = 14, batch_size: int = 2_000) -> None:
    vocab: dict[str, int] = {}
    doc_ids: list[str] = []
    cols, vals, lens, nnz_per_doc = [], [], [], []

    def batches():
        for batch in iter_corpus(batch_size=batch_size, columns=("doc_id", "title", "content")):
            doc_ids.extend(r["doc_id"] for r in batch)
            yield [doc_text(r["title"], r["content"]) for r in batch]

    n_done = 0
    with ProcessPoolExecutor(workers) as ex, tqdm(desc="tokenize", unit="doc") as bar:
        for local_vocab, doc_ptr, term_ids, tfs, lengths in _bounded_map(ex, _count_batch, batches(), 2 * workers):
            remap = np.fromiter((vocab.setdefault(t, len(vocab)) for t in local_vocab), dtype=np.int32, count=len(local_vocab))
            nnz_per_doc.append(np.diff(doc_ptr))
            cols.append(remap[term_ids])
            vals.append(tfs)
            lens.append(lengths)
            n_done += len(lengths)
            bar.update(len(lengths))

    indptr = np.concatenate([[0], np.cumsum(np.concatenate(nnz_per_doc))])
    tf = sp.csr_matrix((np.concatenate(vals), np.concatenate(cols), indptr), shape=(n_done, len(vocab))).tocsc()
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "tf.npz", tf, compressed=False)
    np.save(out_dir / "doc_len.npy", np.concatenate(lens))
    (out_dir / "doc_ids.txt").write_text("\n".join(doc_ids) + "\n", encoding="utf8")
    (out_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf8")


@dataclass
class SparseIndex:
    tf: sp.csc_matrix
    doc_len: np.ndarray
    doc_ids: list[str]
    vocab: dict[str, int]

    @classmethod
    def load(cls, path: Path = INDEX_DIR) -> "SparseIndex":
        return cls(
            tf=sp.load_npz(path / "tf.npz").tocsc(),
            doc_len=np.load(path / "doc_len.npy"),
            doc_ids=(path / "doc_ids.txt").read_text(encoding="utf8").split("\n")[:-1],
            vocab={t: i for i, t in enumerate((path / "vocab.txt").read_text(encoding="utf8").split("\n")[:-1])},
        )

    def query_terms(self, text: str) -> list[int]:
        """Analyzed query as term ids, repeats kept (Lucene scores a repeated term once per occurrence), OOV dropped."""
        return [self.vocab[t] for t in tokenize(text) if t in self.vocab]
