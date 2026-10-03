"""Paper-faithful chunking for dense retrieval.

`chunk_text` is copied verbatim from the harness (src/scripts/answer_generation/index_document_vectors.py
@ d36685e2, MIT) so it runs on the embedding box without the harness's qdrant/openai imports.
"""

import tiktoken

_encoding = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 512


def chunk_text(text: str, chunk_size: int, overlap_frac: float = 0.1) -> list[str]:
    """Split *text* into token-based chunks with configurable overlap."""
    tokens = _encoding.encode(text, disallowed_special=())
    if len(tokens) <= chunk_size:
        return [text]
    overlap = max(1, int(chunk_size * overlap_frac))
    stride = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(tokens), stride):
        chunk_tokens = tokens[start : start + chunk_size]
        chunks.append(_encoding.decode(chunk_tokens))
        if start + chunk_size >= len(tokens):
            break
    return chunks
