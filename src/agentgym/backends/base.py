"""The inference abstraction the harness talks to.

One interface, several implementations (vLLM, SGLang, any OpenAI-compatible endpoint).
The agent code never learns which engine is underneath, which is what makes the serving
comparison a fair one: identical prompts, identical tool schemas, identical parsing.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    finish_reason: str = "stop"
    backend: str = ""
    model: str = ""


class InferenceBackend(ABC):
    """Generate one assistant turn, optionally with tool calls."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        adapter: str | None = None,
    ) -> GenerationResult:
        ...

    def health_check(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class BackendStats:
    """Rolling counters used by the serving benchmarks."""

    requests: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def record(self, result: GenerationResult) -> None:
        self.requests += 1
        self.prompt_tokens += result.prompt_tokens
        self.completion_tokens += result.completion_tokens
        self.latencies_ms.append(result.latency_ms)

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        idx = min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1))))
        return round(ordered[idx], 2)

    def summary(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_p50_ms": self.percentile(50),
            "latency_p95_ms": self.percentile(95),
            "latency_mean_ms": round(
                sum(self.latencies_ms) / len(self.latencies_ms), 2
            ) if self.latencies_ms else 0.0,
        }


class Timer:
    def __enter__(self) -> Timer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000


__all__ = ["InferenceBackend", "GenerationResult", "BackendStats", "Timer"]
