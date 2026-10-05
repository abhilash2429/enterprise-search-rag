"""Answer verifier: a second model checks each cited claim against the context the answerer saw.

Run after generation, on the answer with its [n] markers and the same numbered documents. The verdict feeds a
confidence flag (flag only: the answer is still shown). Prompt written once, not tuned on dev.
"""

import json
from dataclasses import dataclass

from entsearch.answerer import MISSING_PREFIX, format_context, version_note

MODEL = "gpt-6-luna"
# Reranker top score below this also flags an answer. Picked once on dev (runs/rerank_router_d100/flagger.json):
# max F1 for judge-wrong answers, with the verifier verdict OR'd in.
FLAG_TAU = 0.9489
VERDICTS = ("supported", "partially_supported", "unsupported")
CLAIM_VERDICTS = ("supported", "partially_supported", "unsupported", "contradicted")  # luna also grades compound claims partially

PROMPT = """You check an answer written by a question-answering system against the documents it was given. The \
documents come from an imperfect retrieval system, so many of them may be irrelevant.

Steps:
1. Split the answer into its factual claims. A final sentence starting with "{missing_prefix}" only says what is \
missing; it is not a claim.
2. For each claim, check the documents it cites, or all documents if it cites none. Verdict "supported" if a document \
states it, "unsupported" if no document states it, "contradicted" if a document states something different. Where \
documents are versions of each other, an answer that gives the newer value is not contradicted by the older one.
3. Check whether the answer addresses the question as asked: the right entity, project, person, time period and \
scope. An answer can be fully supported by the documents and still be about the wrong thing.

Overall verdict:
- "supported": every claim is supported and the answer addresses the question.
- "unsupported": the claim that answers the question is unsupported or contradicted, or the answer does not address \
the question.
- "partially_supported": anything else.

Reply with only this JSON:
{{"claims": [{{"claim": "...", "cited": [1], "verdict": "supported"}}], "addresses_question": true, \
"verdict": "supported", "reasoning": "Two or three sentences: the main problem, or why the answer holds up."}}

## Documents
{context}
{version_note}
## Question
{question}

## Answer to check
{answer}
"""


def build_prompt(
    question: str, answer: str, doc_ids: list[str], docs: dict[str, tuple[str, str, str]], pairs: list[tuple[str, str, float]]
) -> str:
    """answer: the answerer's text with its [n] markers; doc_ids/docs/pairs: exactly the context it was given."""
    return PROMPT.format(
        missing_prefix=MISSING_PREFIX, context=format_context(doc_ids, docs), version_note=version_note(doc_ids, pairs),
        question=question, answer=answer,
    )


@dataclass
class Verdict:
    verdict: str
    addresses_question: bool
    claims: list[dict]
    reasoning: str

    @property
    def flagged(self) -> bool:
        return self.verdict != "supported" or not self.addresses_question


def parse(text: str) -> Verdict:
    """The JSON object in the reply (code fences tolerated). Raises ValueError on anything malformed."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in verifier reply: {text[:200]!r}")
    try:
        d = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"bad JSON in verifier reply: {e}") from e
    if d.get("verdict") not in VERDICTS:
        raise ValueError(f"verdict {d.get('verdict')!r} not in {VERDICTS}")
    if not isinstance(d.get("addresses_question"), bool):
        raise ValueError("addresses_question must be a boolean")
    claims = d.get("claims")
    if not isinstance(claims, list) or any(c.get("verdict") not in CLAIM_VERDICTS for c in claims):
        raise ValueError("claims must be a list of {claim, cited, verdict}")
    return Verdict(d["verdict"], d["addresses_question"], claims, str(d.get("reasoning", "")))
