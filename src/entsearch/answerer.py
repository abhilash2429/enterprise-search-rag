"""Cited answerer: numbered context, [n] citations per claim, version-pair flags, explicit refusal.

The judge sees the answer with [n] markers stripped (`Parsed.text`); the cited text and cited doc ids are kept for
the demo and for cited-doc precision. Docs come in reranked order and are numbered 1..len(ids).
"""

import re
from dataclasses import dataclass

REFUSAL = "The provided documents do not contain the information needed to answer this question."
MISSING_PREFIX = "Not covered by the documents:"

PROMPT = """You answer questions about a company using only the documents below. They come from an imperfect retrieval \
system, so many of them may be irrelevant.

Rules:
- Use only facts stated in the documents. Do not guess or add outside knowledge.
- Cite every claim with the number of the document that supports it, in square brackets right after the claim, \
like [2] or [2][5].
- Be concise: answer the question directly, then add only details that are directly relevant. Write plain text, \
no markdown formatting.
- If documents disagree, for example an earlier proposal and a later decision, or two versions of the same document, \
give the most recent or applied value as the answer, and also state the earlier value. Cite both.
- If the documents answer only part of the question, answer that part, then add one sentence starting with \
"{missing_prefix}" naming what is not covered.
- If the documents contain nothing that answers the question, reply with exactly this sentence and nothing else: \
"{refusal}"

## Documents
{context}
{version_note}
## Question
{question}

## Answer
"""

# [2], [2, 5], and gpt-oss's native 【2】 / 【3†L1-L4】, which it falls back to despite the instruction.
_CITE = re.compile(r"\s*[\[【](\d+(?:\s*[,;]\s*\d+)*)(?:†[^\]】]*)?[\]】]")


def format_context(doc_ids: list[str], docs: dict[str, tuple[str, str, str]]) -> str:
    """docs: doc_id -> (source_type, title, content)."""
    return "\n\n".join(
        f"--- Document [{i}] (source: {docs[d][0]}) ---\nTitle: {docs[d][1]}\n\n{docs[d][2]}" for i, d in enumerate(doc_ids, 1)
    )


def version_note(doc_ids: list[str], pairs: list[tuple[str, str, float]]) -> str:
    if not pairs:
        return ""
    num = {d: i for i, d in enumerate(doc_ids, 1)}
    listed = "; ".join(f"[{num[a]}] and [{num[b]}]" for a, b, _ in pairs)
    return (
        f"\nNote: these documents look like versions of the same document, so differences between them may be "
        f"updates: {listed}.\n"
    )


def build_prompt(question: str, doc_ids: list[str], docs: dict[str, tuple[str, str, str]], pairs: list[tuple[str, str, float]]) -> str:
    return PROMPT.format(
        missing_prefix=MISSING_PREFIX, refusal=REFUSAL, context=format_context(doc_ids, docs),
        version_note=version_note(doc_ids, pairs), question=question,
    )


@dataclass
class Parsed:
    text: str  # markers stripped, what the judge sees
    cited_doc_ids: list[str]  # in order of first citation
    invalid_citations: int  # [n] outside 1..len(doc_ids)
    abstained: bool  # exact full refusal
    partial: bool  # answered with a "not covered" caveat


def parse(answer: str, doc_ids: list[str]) -> Parsed:
    cited, invalid = [], 0
    for m in _CITE.finditer(answer):
        for n in map(int, re.split(r"[,;]", m.group(1))):
            if 1 <= n <= len(doc_ids):
                cited.append(doc_ids[n - 1])
            else:
                invalid += 1
    text = _CITE.sub("", answer).strip()
    return Parsed(
        text=text, cited_doc_ids=list(dict.fromkeys(cited)), invalid_citations=invalid,
        abstained=text.strip().strip('"') == REFUSAL, partial=MISSING_PREFIX in text,
    )
