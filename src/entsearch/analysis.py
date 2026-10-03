"""Approximation of Lucene's `standard` analyzer: UAX#29 word segmentation, lowercase, no stopwords, no stemming."""

import regex

_BOUNDARY = regex.compile(r"\b", flags=regex.WORD | regex.V1)
_WORDLIKE = regex.compile(r"[\p{L}\p{N}\p{Extended_Pictographic}]")
MAX_TOKEN_LEN = 255
# regex's UAX#29 keeps a word-initial apostrophe/MidLetter char attached ("'etcd"); Lucene drops it.
_LEADING_MID = "'’.:,;·"


def tokenize(text: str) -> list[str]:
    out = []
    for seg in _BOUNDARY.split(text):
        if not _WORDLIKE.search(seg):
            continue
        seg = seg.lstrip(_LEADING_MID).lower()
        # StandardTokenizer splits over-long tokens at max_token_length.
        out.extend(seg[i : i + MAX_TOKEN_LEN] for i in range(0, len(seg), MAX_TOKEN_LEN))
    return out
