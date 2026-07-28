"""Serving benchmarks for multi-turn agent workloads.

The interesting question is not "which engine is faster on a generic prompt". It is:
**agent rollouts share a long, growing prefix -- the system prompt, the tool schemas, and
every previous turn -- so how much does prefix caching actually buy, and where does it
stop helping?**

Four experiments:

  prefix_scaling      latency vs conversation depth at several concurrencies. If prefix
                      caching works, per-turn latency should stay roughly flat as depth
                      grows, instead of climbing with prompt length.
  concurrency         throughput and p95 as concurrent agents increase.
  structured_output   how often the model emits a parseable, schema-valid tool call.
  multi_lora          cost of serving base and adapters from one engine.

Results are written as JSON for the report; nothing here judges model quality.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentgym.backends.base import InferenceBackend
from agentgym.backends.openai_compat import VLLMBackend
from agentgym.harness.prompts import system_prompt
from agentgym.mcp import tools as tools_mod

# A representative tool observation: long, JSON, and highly repetitive across turns --
# exactly the shape that makes prefix caching matter.
_OBSERVATION = json.dumps([
    {"name": f"api-service-7bb7ddc7b4-{i:05x}", "phase": "Running",
     "reason": None, "ready": True, "restarts": 0,
     "node": f"node-cpu-{i % 2 + 1}", "labels": {"app": "api-service"}}
    for i in range(12)
], indent=2)


def _conversation(turns: int) -> list[dict[str, str]]:
    """A synthetic agent conversation `turns` deep, with a realistic shared prefix.

    The system message carries the full 36-tool catalog, exactly as the harness sends it.
    That catalog is several thousand tokens and is byte-identical on every request, so it
    is the single largest thing prefix caching can reuse -- benchmarking against a bare
    system prompt would understate the effect being measured.
    """
    messages: list[dict[str, str]] = [
        {"role": "system",
         "content": system_prompt([t.openai_schema() for t in tools_mod.REGISTRY.values()])},
        {"role": "user", "content":
            "INCIDENT: service degradation in namespace `inc-00042`.\n"
            "Application: web-api. Entrypoint: edge-proxy.\n"
            "Diagnose and fix it."},
    ]
    for i in range(turns):
        messages.append({"role": "assistant", "content":
                         f"Checking pod state (step {i}).\n\n"
                         '```json\n{"name": "list_pods", '
                         '"arguments": {"namespace": "inc-00042"}}\n```'})
        messages.append({"role": "user", "content": f"TOOL RESULT:\n{_OBSERVATION}"})
    return messages


@dataclass
class BenchResult:
    name: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, **row: Any) -> None:
        self.rows.append(row)


def _time_call(backend: InferenceBackend, messages: list[dict[str, str]],
               tools: list[dict] | None = None, max_tokens: int = 96) -> tuple[float, int, int]:
    # `tools` is accepted for signature symmetry but deliberately not forwarded: the
    # harness puts the catalog in the system prompt and sends no tools field, so
    # forwarding one here would benchmark a request shape production never issues.
    t0 = time.perf_counter()
    result = backend.generate(messages, None, temperature=0.0, max_tokens=max_tokens)
    return ((time.perf_counter() - t0) * 1000,
            result.prompt_tokens, result.completion_tokens)


def prefix_scaling(backend: InferenceBackend, tools: list[dict],
                   depths=(2, 5, 10, 20), concurrencies=(1, 8, 32),
                   repeats: int = 3) -> BenchResult:
    """Per-turn latency as the shared prefix grows."""
    out = BenchResult("prefix_scaling")
    for depth in depths:
        messages = _conversation(depth)
        for concurrency in concurrencies:
            latencies: list[float] = []
            prompt_tokens = 0
            t0 = time.perf_counter()
            for _ in range(repeats):
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(_time_call, backend, messages, tools)
                               for _ in range(concurrency)]
                    for f in futures:
                        latency, ptoks, _ = f.result()
                        latencies.append(latency)
                        prompt_tokens = ptoks
            wall = time.perf_counter() - t0
            out.add(
                turns=depth, concurrency=concurrency,
                prompt_tokens=prompt_tokens,
                latency_p50_ms=round(statistics.median(latencies), 1),
                latency_p95_ms=round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1),
                latency_mean_ms=round(statistics.mean(latencies), 1),
                requests_per_second=round(len(latencies) / wall, 2),
                # The headline number: milliseconds of latency per thousand prompt
                # tokens. Flat across depth means the prefix is genuinely being reused.
                ms_per_1k_prompt_tokens=round(
                    statistics.median(latencies) / max(1, prompt_tokens / 1000), 1),
            )
            print(f"  depth={depth:3d} conc={concurrency:3d} "
                  f"p50={out.rows[-1]['latency_p50_ms']:8.1f}ms "
                  f"ptoks={prompt_tokens:6d} "
                  f"rps={out.rows[-1]['requests_per_second']:6.2f}", flush=True)
    return out


def structured_output(backend: InferenceBackend, tools: list[dict],
                      samples: int = 40) -> BenchResult:
    """How reliably the served model emits a valid, dispatchable tool call."""
    from jsonschema import Draft202012Validator, ValidationError

    from agentgym.harness.llm_policy import LLMPolicy

    out = BenchResult("structured_output")
    policy = LLMPolicy(backend=backend)
    registry = {t.name: t for t in tools_mod.REGISTRY.values()}
    messages = _conversation(3)

    parsed = known = valid = truncated = 0
    for _ in range(samples):
        # 200 tokens is not enough for a reasoning model to finish thinking and then
        # emit JSON; a smaller cap measures truncation rather than structured-output
        # ability. Matches the harness budget so the number means the same thing.
        result = backend.generate(messages, None, temperature=0.7, max_tokens=1536)
        if result.finish_reason == "length":
            truncated += 1
        action = policy._to_action(result.text, result.tool_calls)
        name = getattr(action, "name", "submit_diagnosis")
        args = getattr(action, "args", {})
        if name != "__unparseable__":
            parsed += 1
        tool = registry.get(name)
        if tool is not None:
            known += 1
            try:
                Draft202012Validator(tool.parameters).validate(args)
                valid += 1
            except ValidationError:
                pass
    out.add(samples=samples,
            truncated_rate=round(truncated / samples, 4),
            parse_rate=round(parsed / samples, 4),
            known_tool_rate=round(known / samples, 4),
            schema_valid_rate=round(valid / samples, 4))
    return out


def multi_lora(base_url: str, model: str, adapters: list[str],
               samples: int = 20) -> BenchResult:
    """Latency of the base model vs adapters served from the same engine."""
    out = BenchResult("multi_lora")
    tools = [t.openai_schema() for t in tools_mod.REGISTRY.values()]
    messages = _conversation(3)
    for target in [model, *adapters]:
        backend = VLLMBackend(base_url=base_url, model=target)
        latencies = []
        for _ in range(samples):
            try:
                latency, _, _ = _time_call(backend, messages, tools)
                latencies.append(latency)
            except Exception as exc:
                out.add(target=target, error=str(exc)[:200])
                break
        if latencies:
            out.add(target=target, samples=len(latencies),
                    latency_p50_ms=round(statistics.median(latencies), 1),
                    latency_mean_ms=round(statistics.mean(latencies), 1))
        backend.close()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="AgentGym serving benchmarks")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--backend-name", default="vllm")
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip", nargs="*", default=[])
    args = p.parse_args()

    backend = VLLMBackend(base_url=args.base_url, model=args.model)
    if not backend.wait_until_ready(timeout_s=1200):
        raise SystemExit(f"engine at {args.base_url} never became ready")

    tools = [t.openai_schema() for t in tools_mod.REGISTRY.values()]
    report: dict[str, Any] = {
        "backend": args.backend_name, "model": args.model,
        "tool_count": len(tools),
    }

    if "prefix" not in args.skip:
        print("== prefix scaling ==", flush=True)
        report["prefix_scaling"] = prefix_scaling(backend, tools).rows
    if "structured" not in args.skip:
        print("== structured output ==", flush=True)
        report["structured_output"] = structured_output(backend, tools).rows
    if args.adapters and "lora" not in args.skip:
        print("== multi-LoRA ==", flush=True)
        report["multi_lora"] = multi_lora(args.base_url, args.model, args.adapters).rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
