from types import SimpleNamespace

import pytest

from entsearch.llm import completed_usage


def test_raw_dict_missing_sdk_required_detail_fields():
    # Bedrock payload seen 2026-10-04: no input_tokens_details.cache_write_tokens, which the SDK schema requires.
    ev = SimpleNamespace(response={"usage": {
        "input_tokens": 14512, "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 300, "output_tokens_details": {"reasoning_tokens": 120},
    }})
    u = completed_usage(ev)
    assert (u.input_tokens, u.output_tokens, u.reasoning_tokens) == (14512, 300, 120)


def test_parsed_sdk_object():
    usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 5, "output_tokens": 7, "output_tokens_details": {"reasoning_tokens": 2}})
    u = completed_usage(SimpleNamespace(response=SimpleNamespace(usage=usage)))
    assert (u.input_tokens, u.output_tokens, u.reasoning_tokens) == (5, 7, 2)


def test_missing_core_counts_raise():
    with pytest.raises(RuntimeError):
        completed_usage(SimpleNamespace(response={"usage": {"input_tokens": 5}}))
    with pytest.raises(RuntimeError):
        completed_usage(SimpleNamespace(response={"usage": None}))
