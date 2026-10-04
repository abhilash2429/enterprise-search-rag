from types import SimpleNamespace

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from entsearch.retrieval.rerank import _sdpa_repeat_kv


def test_matches_transformers_sdpa_with_and_without_mask():
    torch.manual_seed(0)
    module = SimpleNamespace(num_key_value_groups=2, is_causal=True)
    q, k, v = torch.randn(2, 4, 6, 8), torch.randn(2, 2, 6, 8), torch.randn(2, 2, 6, 8)
    causal = torch.ones(6, 6, dtype=torch.bool).tril()
    padded = causal.expand(2, 1, 6, 6).clone()
    padded[1, :, :, :2] = False  # left padding on the second sequence
    padded[1, :, :2, :2] = torch.eye(2, dtype=torch.bool)  # pad rows attend to themselves, avoiding all-masked rows
    for mask in (None, padded):
        ours, _ = _sdpa_repeat_kv(module, q, k, v, mask, scaling=0.3)
        ref, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=0.3)
        torch.testing.assert_close(ours, ref)
