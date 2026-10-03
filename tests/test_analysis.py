from entsearch.analysis import tokenize

# Expected outputs captured from OpenSearch 2.19.1 `standard` analyzer (_analyze).
CASES = {
    "© 2026 Acme® ™ ok": ["©", "2026", "acme", "®", "™", "ok"],
    "ship it ✅👍🏽 👨‍👩‍👧 done ↔ x": ["ship", "it", "✅", "👍🏽", "👨‍👩‍👧", "done", "↔", "x"],
    "❤️ love": ["❤️", "love"],
    "'etcd' config": ["etcd", "config"],
    "Can't stop U.S.A. 1,000.50 v2.3.1 INT-7832": ["can't", "stop", "u.s.a", "1,000.50", "v2.3.1", "int", "7832"],
}


def test_matches_lucene_standard():
    for text, expected in CASES.items():
        assert tokenize(text) == expected, text
