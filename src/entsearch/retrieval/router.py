"""Source router: an LLM guesses which of the 9 sources hold the answer. Used as a soft boost (extra RRF lists
restricted to the predicted sources), never as a hard filter, so a wrong guess costs rank, not recall."""

import json
import re

SOURCES = {
    "confluence": "wiki pages: playbooks, PRDs, runbooks, policies, design docs",
    "jira": "tickets: IT, security, and support incidents with root causes and resolutions",
    "linear": "engineering issues and project tasks",
    "github": "pull requests and code changes",
    "slack": "team chat channel threads",
    "gmail": "email, mostly with customers, partners, and vendors",
    "google_drive": "docs, sheets, and notes: plans, backlogs, handoff notes",
    "hubspot": "CRM records for customer and prospect companies",
    "fireflies": "meeting transcripts",
}
MAX_SOURCES = 3

PROMPT = """You route questions about a company's internal documents to the systems most likely to contain the answer.

Systems:
{systems}

Question: {question}

Return a JSON array of 1 to {max_sources} system names, most likely first. Name a second or third system only if it is \
plausibly where the answer lives. Output only the JSON array."""


def build_prompt(question: str) -> str:
    systems = "\n".join(f"- {k}: {v}" for k, v in SOURCES.items())
    return PROMPT.format(systems=systems, question=question, max_sources=MAX_SOURCES)


def parse(text: str) -> list[str]:
    """Known source names in the model's order, deduped, at most MAX_SOURCES. Empty if nothing parses."""
    m = re.search(r"\[.*?\]", text, re.S)
    try:
        names = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        names = re.findall(r"[a-z_]+", m.group(0)) if m else []
    out = [n for n in dict.fromkeys(str(n).strip().lower() for n in names) if n in SOURCES]
    return out[:MAX_SOURCES]
