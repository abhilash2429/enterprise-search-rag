import random

from entsearch.harness import harness_import
from entsearch.index.chunking import chunk_text


def test_matches_harness_chunker():
    ref = harness_import("src.scripts.answer_generation.index_document_vectors").chunk_text
    rng = random.Random(0)
    words = ["alpha", "Beta", "γάμμα", "日本", "🎉", "x" * 40, "\n\n", "<|endoftext|>", "INT-7832"]
    for n in [0, 1, 50, 511, 512, 513, 2000, 5000]:
        text = " ".join(rng.choice(words) for _ in range(n))
        assert chunk_text(text, 512) == ref(text, 512)
