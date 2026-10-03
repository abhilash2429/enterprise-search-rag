"""Paper BM25 reproduction: index the corpus into OpenSearch (harness settings) and retrieve top-10 for all 500 questions."""
import argparse
import json

from tqdm import tqdm

from entsearch.data import ROOT, load_questions
from entsearch.retrieval import opensearch

p = argparse.ArgumentParser()
p.add_argument("--reindex", action="store_true")
args = p.parse_args()

os_client = opensearch.client()
if args.reindex or not os_client.indices.exists(index=opensearch.INDEX):
    opensearch.build_index(os_client)

run_dir = ROOT / "runs/bm25_opensearch"
run_dir.mkdir(parents=True, exist_ok=True)
with open(run_dir / "retrieval.jsonl", "w", encoding="utf8") as f:
    for row in tqdm(load_questions().itertuples(), total=500, desc="search"):
        f.write(json.dumps({"question_id": row.question_id, "document_ids": opensearch.search(os_client, row.question)}) + "\n")
