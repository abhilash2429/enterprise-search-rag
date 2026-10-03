"""Bedrock (mantle, OpenAI-compatible) client with a disk cache keyed on the full request."""

import hashlib
import json
from dataclasses import dataclass

from aws_bedrock_token_generator import provide_token
from openai import OpenAI
from openai.types.responses import Response

from entsearch.data import ROOT

REGION = "us-east-1"
BASE_URL = f"https://bedrock-mantle.{REGION}.api.aws/v1"
MODEL = "openai.gpt-oss-120b"
CACHE = ROOT / "cache/llm"
# Standard tier, USD per 1M tokens.
PRICE = {MODEL: (0.15, 0.60)}


def bedrock_token() -> str:
    return provide_token(region=REGION)


def client() -> OpenAI:
    return OpenAI(base_url=BASE_URL, api_key=bedrock_token(), max_retries=5, timeout=600)


@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cached: bool

    @property
    def cost(self) -> float:
        p_in, p_out = PRICE[MODEL]
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1e6


def completed_response(ev) -> Response:
    """The SDK leaves `response` as a raw dict when the payload fails its schema. Validate it so a
    malformed payload raises with the offending field instead of an opaque AttributeError."""
    resp = ev.response
    if isinstance(resp, dict):
        try:
            return Response.model_validate(resp)
        except Exception as e:
            raise RuntimeError(f"malformed response.completed payload: {e}; raw={json.dumps(resp)[:2000]}") from e
    return resp


def generate(llm: OpenAI, prompt: str, model: str = MODEL, effort: str = "medium") -> Generation:
    """Same call shape as the harness: Responses API, raw stream, reasoning effort, reasoning summary on."""
    key = hashlib.sha256(json.dumps([model, effort, prompt]).encode()).hexdigest()
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
            usage = completed_response(ev).usage
    if usage is None:
        raise RuntimeError("stream ended without response.completed")

    gen = Generation(
        text="".join(text).strip(),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        cached=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {k: v for k, v in gen.__dict__.items() if k != "cached"}
    path.write_text(json.dumps(rec), encoding="utf8")
    return gen
