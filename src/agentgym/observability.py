"""Tracing and metrics for the agent loop.

One OpenTelemetry span per episode, one child span per tool call, plus Prometheus
counters and histograms. Both degrade to no-ops when the libraries or an endpoint are
absent, so a rollout on a compute node with no collector behaves exactly as it does under
a full observability stack -- instrumentation must never be a reason an episode fails.

The span attributes are chosen to answer the questions you actually ask during a
postmortem of an agent run: which tool, which tier, did it validate, was it unsafe, how
long did the environment take versus the model.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

# --- OpenTelemetry (optional) ------------------------------------------------

_TRACER: Any = None
_OTEL_READY = False


def init_tracing(service_name: str = "agentgym") -> bool:
    """Configure a tracer if an OTLP endpoint is set. Returns True when active."""
    global _TRACER, _OTEL_READY
    if _OTEL_READY:
        return _TRACER is not None
    _OTEL_READY = True

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("agentgym")
        return True
    except Exception:
        _TRACER = None
        return False


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """A span if tracing is configured, otherwise a zero-cost no-op."""
    init_tracing()
    if _TRACER is None:
        yield None
        return
    with _TRACER.start_as_current_span(name) as sp:
        for key, value in attributes.items():
            if value is not None:
                sp.set_attribute(f"agentgym.{key}", value)
        yield sp


def set_attributes(sp: Any, **attributes: Any) -> None:
    if sp is None:
        return
    for key, value in attributes.items():
        if value is not None:
            with contextlib.suppress(Exception):
                sp.set_attribute(f"agentgym.{key}", value)


# --- Prometheus (optional) ---------------------------------------------------

class _NoopMetric:
    def labels(self, *a: Any, **k: Any) -> _NoopMetric:
        return self

    def inc(self, *a: Any, **k: Any) -> None: ...
    def observe(self, *a: Any, **k: Any) -> None: ...
    def set(self, *a: Any, **k: Any) -> None: ...


def _make_metrics() -> dict[str, Any]:
    try:
        from prometheus_client import Counter, Histogram
    except Exception:
        return {k: _NoopMetric() for k in
                ("tool_calls", "tool_latency", "episodes", "episode_latency",
                 "episode_reward", "unsafe_actions")}

    return {
        "tool_calls": Counter(
            "agentgym_tool_calls_total",
            "Tool calls by name, tier, and outcome.",
            ["tool", "tier", "outcome"],
        ),
        "tool_latency": Histogram(
            "agentgym_tool_latency_seconds",
            "Tool call latency.",
            ["tool", "tier"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        ),
        "episodes": Counter(
            "agentgym_episodes_total",
            "Episodes by fault family, split, and whether they were resolved.",
            ["fault", "split", "resolved"],
        ),
        "episode_latency": Histogram(
            "agentgym_episode_seconds",
            "Wall time per episode.",
            ["fault"],
            buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300),
        ),
        "episode_reward": Histogram(
            "agentgym_episode_reward",
            "Verified reward per episode.",
            ["fault"],
            buckets=(-2, -1, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
        ),
        "unsafe_actions": Counter(
            "agentgym_unsafe_actions_total",
            "Actions the incident's safety policy forbids.",
            ["tool", "fault"],
        ),
    }


METRICS = _make_metrics()


def record_tool_call(tool: str, tier: str, ok: bool, error_kind: str | None,
                     duration_ms: float, unsafe: bool, fault: str = "") -> None:
    outcome = "ok" if ok else (error_kind or "error")
    METRICS["tool_calls"].labels(tool=tool, tier=tier, outcome=outcome).inc()
    METRICS["tool_latency"].labels(tool=tool, tier=tier).observe(duration_ms / 1000.0)
    if unsafe:
        METRICS["unsafe_actions"].labels(tool=tool, fault=fault).inc()


def record_episode(fault: str, split: str, resolved: bool, reward: float,
                   wall_seconds: float) -> None:
    METRICS["episodes"].labels(
        fault=fault, split=split, resolved=str(resolved).lower()
    ).inc()
    METRICS["episode_latency"].labels(fault=fault).observe(wall_seconds)
    METRICS["episode_reward"].labels(fault=fault).observe(reward)


def start_metrics_server(port: int | None = None) -> bool:
    """Expose /metrics. Used by the gateway deployment, not by batch rollout."""
    port = port or int(os.environ.get("AGENTGYM_METRICS_PORT", "9095"))
    try:
        from prometheus_client import start_http_server

        start_http_server(port)
        return True
    except Exception:
        return False


__all__ = [
    "span", "set_attributes", "init_tracing", "record_tool_call",
    "record_episode", "start_metrics_server", "METRICS",
]
