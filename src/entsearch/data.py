from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/erb/data/documents/test.parquet"
QUESTIONS = ROOT / "data/erb/data/questions/test.parquet"
SPLITS = ROOT / "data/splits"
HARNESS = ROOT / "third_party/erb"

# Four doc_ids appear twice in the parquet with different documents. The harness keeps one
# per id via generated_data/uuid_index.json; these are the copies it does NOT keep, resolved
# by fetching the harness source files and matching title and content length.
SHADOWED = {
    ("dsid_8a0c5430bac64f8da21c2cee5a7f4df5", 3645),
    ("dsid_feb1e9063ebb4947bb4f935393c01f0f", 5616),
    ("dsid_f292876dbb47462d85997383af306490", 7052),
    ("dsid_6df52fdb96ae4edcb76464738bca3340", 4520),
}


def doc_text(title: str, content: str) -> str:
    return f"{title}\n\n{content}"


def load_questions(split: str | None = None) -> pd.DataFrame:
    q = pd.read_parquet(QUESTIONS)
    for col in ("source_types", "expected_doc_ids", "answer_facts"):
        q[col] = q[col].map(list)
    if split is not None:
        ids = set((SPLITS / f"{split}.txt").read_text().split())
        q = q[q.question_id.isin(ids)].reset_index(drop=True)
    return q


def iter_corpus(batch_size: int = 20_000, columns=("doc_id", "source_type", "title", "content")) -> Iterator[list[dict]]:
    """Yields batches of harness-equivalent documents: one per doc_id, shadowed duplicates removed."""
    cols = list(dict.fromkeys(["doc_id", "content", *columns]))
    for batch in pq.ParquetFile(CORPUS).iter_batches(batch_size=batch_size, columns=cols):
        rows = [
            r for r in batch.to_pylist()
            if (r["doc_id"], len(r["content"])) not in SHADOWED
        ]
        yield [{k: r[k] for k in columns} for r in rows]


def load_docs(doc_ids: list[str], with_source: bool = False) -> dict[str, tuple[str, ...]]:
    """doc_id -> (title, content), or (source_type, title, content) with_source, for the harness-equivalent copy."""
    cols = ["doc_id", "source_type", "title", "content"] if with_source else ["doc_id", "title", "content"]
    t = pq.read_table(CORPUS, columns=cols, filters=pc.field("doc_id").isin(list(set(doc_ids))))
    return {
        r["doc_id"]: tuple(r[c] for c in cols[1:])
        for r in t.to_pylist()
        if (r["doc_id"], len(r["content"])) not in SHADOWED
    }
