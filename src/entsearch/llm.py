"""Bedrock (mantle, OpenAI-compatible) client with a disk cache keyed on the full request."""

import hashlib
import json
from dataclasses import dataclass

from aws_bedrock_token_generator import provide_token
from openai import OpenAI

from entsearch.data import ROOT

REGION = "us-east-1"
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws/v1"
MODEL = "openai.gpt-oss-120b"
CACHE = ROOT / "cache/llm"
# Local bridge to Azure AI Foundry (OpenAI Responses API, no auth on the local port).
BRIDGE_URL = "http://127.0.0.1:4100/v1"
# USD per 1M tokens (input, output). gpt-oss: Bedrock standard tier. gpt-6-luna: Azure, under 272K input tokens.
PRICE = {MODEL: (0.15, 0.60), "gpt-6-luna": (0.125, 0.50)}


def bedrock_token() -> str:
    return provide_token(region=REGION)


def client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=bedrock_token(), max_retries=5, timeout=600)


def bridge_client() -> OpenAI:
    return OpenAI(base_url=BRIDGE_URL, api_key="unused", max_retries=5, timeout=600)


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached: bool
    model: str = MODEL  # cache records written before this field existed are all gpt-oss

    @property
    def cost(self) -> float:
        p_in, p_out = PRICE[self.model]
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1e6


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int


def completed_usage(ev) -> Usage:
    """Token counts from a response.completed event.

    The SDK leaves `response` as a raw dict when the payload fails its schema, and Bedrock does not always send fields
    the SDK marks required (e.g. usage.input_tokens_details.cache_write_tokens). Only the three counts used here are
    required; anything else missing is ignored.
    """
    resp = ev.response
    usage = resp.get("usage") if isinstance(resp, dict) else resp.usage
    if usage is None:
        raise RuntimeError("response.completed has no usage")
    if not isinstance(usage, dict):
        usage = usage.model_dump()
    try:
        return Usage(usage["input_tokens"], usage["output_tokens"], (usage.get("output_tokens_details") or {})["reasoning_tokens"])
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"response.completed usage lacks {e}; usage={json.dumps(usage)[:500]}") from e


def generate(llm: OpenAI, prompt: str, model: str = MODEL, effort: str = "medium", seed: int = 0) -> Generation:
    """Same call shape as the harness: Responses API, raw stream, reasoning effort, reasoning summary on.

    seed is a replicate index for the cache key only (sampling is not seeded server side), so seeds 1, 2, ... are
    fresh samples of the same prompt. Seed 0 keeps the original key.
    """
    key_parts = [model, effort, prompt] + ([seed] if seed else [])
    key = hashlib.sha256(json.dumps(key_parts).encode()).hexdigest()
    path = CACHE / key[:2] / f"{key}.json"
    if path.exists():
        return Generation(**{**json.loads(path.read_text(encoding="utf8")), "cached": True})

    text, usage = [], None
    stream = llm.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
        stream=True,
        reasoning={"effort": effort, "summary": "auto"},
    )
    for ev in stream:
        if ev.type == "response.output_text.delta":
            text.append(ev.delta)
        elif ev.type == "response.completed":
            usage = completed_usage(ev)
    if usage is None:
        raise RuntimeError("stream ended without response.completed")

    gen = Generation(
        text="".join(text).strip(),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        cached=False,
        model=model,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {k: v for k, v in gen.__dict__.items() if k != "cached"}
    path.write_text(json.dumps(rec), encoding="utf8")
    return gen
