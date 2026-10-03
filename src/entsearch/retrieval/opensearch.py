"""Paper-faithful BM25: harness index settings and query, loaded from the parquet instead of the JSON tree."""

from opensearchpy import OpenSearch, helpers
from tqdm import tqdm

from entsearch.data import doc_text, iter_corpus
from entsearch.harness import harness_import

INDEX = "enterpriserag"
URL = "http://localhost:9200"
N_DOCS = 511_958


def client(url: str = URL) -> OpenSearch:
    return OpenSearch(hosts=[url], use_ssl=False, verify_certs=False, timeout=120)


def build_index(os_client: OpenSearch, index: str = INDEX) -> None:
    settings = harness_import("src.scripts.answer_generation.index_document_bm25").INDEX_SETTINGS
    if os_client.indices.exists(index=index):
        os_client.indices.delete(index=index)
    os_client.indices.create(index=index, body=settings)
    # Refresh is disabled only for load speed; it does not touch scoring.
    os_client.indices.put_settings(index=index, body={"index": {"refresh_interval": "-1"}})

    def actions():
        for batch in iter_corpus(columns=("doc_id", "title", "content")):
            for r in batch:
                yield {
                    "_index": index,
                    "_id": r["doc_id"],
                    "_source": {"dataset_doc_uuid": r["doc_id"], "text": doc_text(r["title"], r["content"])},
                }

    ok = 0
    for success, info in tqdm(
        helpers.parallel_bulk(os_client, actions(), chunk_size=500, thread_count=4, raise_on_error=True),
        total=N_DOCS, desc="index",
    ):
        ok += success
    os_client.indices.put_settings(index=index, body={"index": {"refresh_interval": None}})
    os_client.indices.refresh(index=index)
    count = os_client.count(index=index)["count"]
    assert count == N_DOCS, f"indexed {count}, expected {N_DOCS}"


def search(os_client: OpenSearch, query: str, k: int = 10, index: str = INDEX) -> list[str]:
    body = {"query": {"match": {"text": {"query": query}}}, "size": k, "_source": ["dataset_doc_uuid"]}
    hits = os_client.search(index=index, body=body)["hits"]["hits"]
    return [h["_source"]["dataset_doc_uuid"] for h in hits if h["_source"].get("dataset_doc_uuid")]


def analyze(os_client: OpenSearch, text: str, index: str = INDEX) -> list[str]:
    return [t["token"] for t in os_client.indices.analyze(index=index, body={"field": "text", "text": text})["tokens"]]
