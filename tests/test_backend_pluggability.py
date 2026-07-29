"""The serving abstraction's actual claim, asserted.

`backends/base.py` says: *"The agent code never learns which engine is underneath, which is
what makes the serving comparison a fair one: identical prompts, identical tool schemas,
identical parsing."*

That sentence is load-bearing for every cross-engine number the project reports. If the
backends differ in what they send or how they parse a reply, a benchmark comparing them
measures the client rather than the engine.

These run against a stub OpenAI-compatible server, so they need neither vLLM nor SGLang
installed. That is deliberate: the claim is about the interface, and a test for it should not
depend on which engine happens to be available in the environment.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agentgym.backends.openai_compat import (
    APIModelBackend,
    OpenAICompatBackend,
    SGLangBackend,
    VLLMBackend,
)

BACKENDS = [VLLMBackend, SGLangBackend, APIModelBackend]
REPLY = ("investigating\n\n```json\n"
         '{"name": "list_pods", "arguments": {"namespace": "x"}}\n```')
MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "incident"}]


class _Recorder:
    """Stands in for an OpenAI-compatible server and records what each backend sent.

    Installed as an httpx transport rather than by patching a module attribute, so the
    request travels through the backend's real client construction, headers and JSON
    encoding -- the parts a cross-engine comparison depends on being identical.
    """

    def __init__(self):
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": REPLY}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        })


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    real_client = httpx.Client

    def _stubbed(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(rec.handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _stubbed)
    return rec


def test_no_backend_reimplements_the_client():
    """Divergent implementations are exactly how two engines stop being comparable."""
    for b in BACKENDS:
        assert issubclass(b, OpenAICompatBackend)
        ignored = {"name", "_abc_impl"}
        overrides = {k for k in vars(b)
                     if not k.startswith("__") and k not in ignored}
        assert not overrides, (
            f"{b.__name__} overrides {overrides}; it is no longer the same client, so a "
            "benchmark against another backend would measure the client, not the engine")


def test_backends_stay_distinguishable_in_a_results_table():
    names = [b.name for b in BACKENDS]
    assert names == ["vllm", "sglang", "api"]
    assert len(set(names)) == 3


def test_identical_request_payload_across_backends(recorder):
    """The bytes on the wire must not depend on which backend object was constructed."""
    sent = []
    for b in BACKENDS:
        recorder.requests.clear()
        b(base_url="http://x/v1", model="m").generate(
            MESSAGES, temperature=0.0, max_tokens=64)
        assert recorder.requests, f"{b.name} sent no request"
        sent.append(recorder.requests[-1])
    for other, b in zip(sent[1:], BACKENDS[1:], strict=True):
        assert other == sent[0], f"{b.name} sends a different payload than vllm"


def test_identical_parsing_across_backends(recorder):
    """One server reply must produce one set of numbers, whoever asked for it."""
    seen = set()
    for b in BACKENDS:
        r = b(base_url="http://x/v1", model="m").generate(MESSAGES)
        seen.add((r.text, r.prompt_tokens, r.completion_tokens, r.finish_reason))
    assert len(seen) == 1, f"backends parsed one reply differently: {seen}"


def test_result_carries_the_backend_label(recorder):
    """A benchmark row has to be attributable to the engine that produced it."""
    for b in BACKENDS:
        r = b(base_url="http://x/v1", model="m").generate(MESSAGES)
        assert r.backend == b.name
