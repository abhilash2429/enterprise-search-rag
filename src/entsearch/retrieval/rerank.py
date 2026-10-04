"""Qwen3-Reranker cross-encoder: score = p("yes") at the last position, per the model card's template.

Pointwise: each (query, doc) pair is scored independently, so reranking the top-d candidates for any d <= the
scored depth is just a re-sort of a prefix of the cached scores.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import sdpa_mask

MODEL = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCTION = (
    "Given a question about a company's internal documents (email, chat, tickets, wikis, code), "
    "retrieve documents that contain information needed to answer it."
)
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _sdpa_repeat_kv(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """SDPA with K/V expanded to every query head, as transformers' repeat_kv does for masked batches.

    For unmasked batches transformers passes enable_gqa=True instead. Windows torch builds have no FlashAttention and
    the memory-efficient kernel rejects GQA, so those batches fall back to the math kernel, which materializes L x L
    attention: a batch of 4 x 4096 tokens runs a 6 GB GPU out of memory. Expanding K/V keeps every batch on the
    memory-efficient kernel.
    """
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, scale=scaling,
        is_causal=query.shape[2] > 1 and attention_mask is None and is_causal,
    )
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("sdpa_repeat_kv", _sdpa_repeat_kv)
# The mask builder is looked up by the same name; without this the function gets no SDPA-style 4D mask.
AttentionMaskInterface.register("sdpa_repeat_kv", sdpa_mask)


class Reranker:
    def __init__(
        self, instruction: str = INSTRUCTION, max_length: int = 4096, token_budget: int = 16_384, model: str = MODEL,
        dtype: torch.dtype = torch.float16,
    ):
        """max_length caps the full prompt; the doc tail is truncated. token_budget caps padded tokens per batch.

        fp16, not bf16: against fp32, bf16 p(yes) drifts up to 0.06 with batch shape and creates spurious ties
        (9 of 48 pairs); fp16 stays within 0.007 at the same speed.
        """
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, attn_implementation="sdpa_repeat_kv").cuda().eval()
        self.instruction, self.token_budget = instruction, token_budget
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(_PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(_SUFFIX, add_special_tokens=False)
        self.body_len = max_length - len(self.prefix) - len(self.suffix)

    def _ids(self, query: str, doc: str) -> list[int]:
        body = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
        return self.prefix + self.tok.encode(body, add_special_tokens=False)[: self.body_len] + self.suffix

    @torch.inference_mode()
    def score(self, query: str, docs: list[str]) -> np.ndarray:
        """p(yes) per doc, in input order."""
        ids = [self._ids(query, d) for d in docs]
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        out = np.empty(len(ids), dtype=np.float32)
        start = 0
        while start < len(order):
            # Longest first, so the first pair in each batch sets the padded length.
            n = max(1, self.token_budget // len(ids[order[start]]))
            batch = order[start : start + n]
            enc = self.tok.pad({"input_ids": [ids[i] for i in batch]}, padding=True, return_tensors="pt").to("cuda")
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1, [self.no, self.yes]].float()
            out[batch] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            start += n
        return out
