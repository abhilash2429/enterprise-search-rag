from entsearch.answerer import MISSING_PREFIX, REFUSAL, build_prompt, parse

IDS = ["d1", "d2", "d3"]


def test_parse_strips_markers_and_maps_citations():
    p = parse("Rate is 30% [2], earlier 20% [3][1]. Owner is Ana [2, 3].", IDS)
    assert p.text == "Rate is 30%, earlier 20%. Owner is Ana."
    assert p.cited_doc_ids == ["d2", "d3", "d1"]
    assert p.invalid_citations == 0
    assert not p.abstained and not p.partial


def test_parse_gpt_oss_native_citations():
    p = parse("Metric is x【2】. Budget 2 ms. 【3†L1-L4】", IDS)
    assert p.text == "Metric is x. Budget 2 ms."
    assert p.cited_doc_ids == ["d2", "d3"]


def test_parse_counts_out_of_range_citations():
    p = parse("X [4] and Y [0] and Z [1].", IDS)
    assert p.cited_doc_ids == ["d1"] and p.invalid_citations == 2


def test_parse_abstention_and_partial():
    assert parse(REFUSAL, IDS).abstained
    assert parse(f'"{REFUSAL}"', IDS).abstained
    p = parse(f"Budget is 5 RPS [1]. {MISSING_PREFIX} the concurrency values.", IDS)
    assert p.partial and not p.abstained


def test_prompt_numbers_docs_and_flags_versions():
    docs = {d: ("slack", f"title {d}", f"body {d}") for d in IDS}
    prompt = build_prompt("q?", IDS, docs, [("d1", "d3", 0.93)])
    assert "--- Document [3] (source: slack) ---\nTitle: title d3" in prompt
    assert "[1] and [3]" in prompt
    assert "Note:" not in build_prompt("q?", IDS, docs, [])
