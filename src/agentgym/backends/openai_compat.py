"""OpenAI-compatible backend, used for vLLM, SGLang, and hosted APIs.

vLLM and SGLang both expose /v1/chat/completions, so one client covers both. The
subclasses differ only in how the server is launched and which capabilities are
advertised -- that is exactly the point of the comparison.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from agentgym.backends.base import GenerationResult, InferenceBackend


@dataclass
class OpenAICompatBackend(InferenceBackend):
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen/Qwen3-8B"
    api_key: str = "not-needed"
    timeout_s: float = 180.0
    name: str = "openai-compat"
    # Some servers reject an empty tools list; others reject unknown fields.
    supports_tools: bool = True
    max_retries: int = 3

    _client: httpx.Client = field(init=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            timeout=httpx.Timeout(self.timeout_s),
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    # --- lifecycle ----------------------------------------------------------

    def health_check(self) -> bool:
        try:
            r = self._client.get("/models")
            return r.status_code == 200
        except Exception:
            return False

    def wait_until_ready(self, timeout_s: float = 900.0, poll_s: float = 3.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.health_check():
                return True
            time.sleep(poll_s)
        return False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    # --- generation ---------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        adapter: str | None = None,
    ) -> GenerationResult:
        payload: dict[str, Any] = {
            # A LoRA adapter is served as its own model name; this is how multi-LoRA
            # serving is exercised without restarting the engine.
            "model": adapter or self.model,
            "messages": _sanitize(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools and self.supports_tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                r = self._client.post("/chat/completions", json=payload)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server error {r.status_code}", request=r.request, response=r
                    )
                if r.status_code >= 400:
                    # A 4xx is a request problem; retrying unchanged will not help.
                    raise RuntimeError(f"{r.status_code}: {r.text[:400]}")
                data = r.json()
                elapsed = (time.perf_counter() - started) * 1000
                return _parse(data, elapsed, self.name, payload["model"])
            except Exception as exc:  # transport hiccup or 5xx
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"generation failed after {self.max_retries} attempts: {last_error}")


def _sanitize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop keys the OpenAI schema rejects, and keep tool messages well-formed."""
    out = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", "call_0"),
                "content": str(m.get("content", ""))[:8000],
            })
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant",
                                   "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            out.append(msg)
        else:
            out.append({"role": role, "content": str(m.get("content", ""))})
    return out


def _parse(data: dict[str, Any], elapsed_ms: float, backend: str,
           model: str) -> GenerationResult:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}

    calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                # Malformed arguments are a real failure mode. Pass them through so the
                # gateway rejects them and the trajectory records the mistake, rather
                # than silently repairing what the model actually produced.
                args = {"__malformed__": raw_args}
        else:
            args = raw_args or {}
        calls.append({"name": fn.get("name", ""), "arguments": args})

    return GenerationResult(
        text=message.get("content") or "",
        tool_calls=calls,
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        latency_ms=round(elapsed_ms, 2),
        finish_reason=choice.get("finish_reason", "stop"),
        backend=backend,
        model=model,
    )


@dataclass
class VLLMBackend(OpenAICompatBackend):
    name: str = "vllm"


@dataclass
class SGLangBackend(OpenAICompatBackend):
    name: str = "sglang"


@dataclass
class APIModelBackend(OpenAICompatBackend):
    """A hosted model, used only as a teacher / upper bound."""

    name: str = "api"
    base_url: str = field(default_factory=lambda: os.environ.get(
        "AGENTGYM_API_BASE", "https://api.openai.com/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("AGENTGYM_API_KEY", ""))


__all__ = ["OpenAICompatBackend", "VLLMBackend", "SGLangBackend", "APIModelBackend"]
