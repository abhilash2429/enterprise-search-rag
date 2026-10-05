import json

import pytest

from entsearch.verifier import build_prompt, parse

DOCS = {"a": ("slack", "Rollout", "We ship v2 on 3 March."), "b": ("jira", "Ticket", "v2 slipped to 10 March.")}


def test_prompt_numbers_context_like_the_answerer_and_includes_answer_and_version_note():
    p = build_prompt("When does v2 ship?", "10 March [2], earlier 3 March [1].", ["a", "b"], DOCS, [("a", "b", 0.9)])
    assert "--- Document [1] (source: slack) ---" in p and "--- Document [2] (source: jira) ---" in p
    assert "[1] and [2]" in p
    assert p.rstrip().endswith("10 March [2], earlier 3 March [1].")
    assert "{" in p and "{{" not in p


def reply(**kw):
    d = {"claims": [{"claim": "ships 10 March", "cited": [2], "verdict": "supported"}], "addresses_question": True,
         "verdict": "supported", "reasoning": "Doc 2 states it."}
    return json.dumps({**d, **kw})


def test_parse_supported_is_not_flagged_and_tolerates_fences():
    v = parse("```json\n" + reply() + "\n```")
    assert v.verdict == "supported" and not v.flagged and v.claims[0]["cited"] == [2]


def test_parse_accepts_partially_supported_claims():
    v = parse(reply(claims=[{"claim": "x and y", "cited": [1], "verdict": "partially_supported"}], verdict="partially_supported"))
    assert v.flagged and v.claims[0]["verdict"] == "partially_supported"


@pytest.mark.parametrize("kw", [{"verdict": "partially_supported"}, {"verdict": "unsupported"}, {"addresses_question": False}])
def test_flagged_when_not_fully_supported_or_off_question(kw):
    assert parse(reply(**kw)).flagged


@pytest.mark.parametrize("text", ["no json here", "{not json}", reply(verdict="maybe"), reply(addresses_question="yes"),
                                  reply(claims=[{"claim": "x", "verdict": "probably"}])])
def test_parse_rejects_malformed(text):
    with pytest.raises(ValueError):
        parse(text)
