"""Random access to documents by doc_id for serving: a SQLite file built once from the corpus parquet.

load_docs() in data.py scans the 512K-row parquet with a filter on every call, which is fine for batch runs over
hundreds of questions but costs seconds and gigabytes per query. Build with `python -m entsearch.docstore`.
"""

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd

from entsearch.data import ROOT, iter_corpus

PATH = ROOT / "data/index/docstore.sqlite"


def build(out: Path = PATH) -> int:
    """Writes the harness-equivalent corpus (shadowed duplicates removed) to `out`; returns the row count."""
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute("CREATE TABLE docs (doc_id TEXT PRIMARY KEY, source_type TEXT, title TEXT, content TEXT)")
    n = 0
    for batch in iter_corpus(columns=("doc_id", "source_type", "title", "content")):
        con.executemany("INSERT INTO docs VALUES (:doc_id, :source_type, :title, :content)", batch)
        n += len(batch)
    con.commit()
    con.close()
    tmp.replace(out)
    return n


class DocStore:
    def __init__(self, path: Path = PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; build it with `python -m entsearch.docstore`")
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        # One read-only connection per call: tools may run on different threads. (A sqlite3 connection's own context
        # manager only commits, so callers wrap it in closing().)
        return sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)

    def get(self, doc_ids: list[str]) -> dict[str, tuple[str, str, str]]:
        """doc_id -> (source_type, title, content); unknown ids are absent."""
        if not doc_ids:
            return {}
        with closing(self._connect()) as con:
            rows = con.execute(
                f"SELECT doc_id, source_type, title, content FROM docs WHERE doc_id IN ({','.join('?' * len(doc_ids))})",
                list(doc_ids),
            ).fetchall()
        return {r[0]: (r[1], r[2], r[3]) for r in rows}

    def sources(self) -> pd.Series:
        """source_type (categorical) indexed by doc_id, for every document. A plain dict of 512K pairs costs ~0.7 GB."""
        with closing(self._connect()) as con:
            df = pd.read_sql_query("SELECT doc_id, source_type FROM docs", con)
        return df.set_index("doc_id").source_type.astype("category")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else PATH
    print(f"wrote {build(out):,} documents to {out}")
