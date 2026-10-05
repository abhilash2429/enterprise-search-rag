from entsearch.retrieval.router import MAX_SOURCES, SOURCES, build_prompt, parse


def test_parse_keeps_order_drops_unknown_and_dupes():
    assert parse('["jira", "Confluence", "jira", "notion"]') == ["jira", "confluence"]


def test_parse_tolerates_prose_and_bad_json():
    assert parse('Likely: ["github", "linear"] because...') == ["github", "linear"]
    assert parse("[github, linear]") == ["github", "linear"]
    assert parse("no idea") == []


def test_parse_caps_length():
    assert len(parse(str(list(SOURCES)).replace("'", '"'))) == MAX_SOURCES


def test_prompt_lists_every_source():
    p = build_prompt("q?")
    assert all(f"- {s}:" in p for s in SOURCES) and "Question: q?" in p
