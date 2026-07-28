"""Schema-shift and transport perturbations.

Every agent demo can call a tool that exists. Far fewer keep working when the tool is
renamed, gains a required argument, or starts returning truncated JSON. This module makes
that failure mode measurable by transforming the toolset an agent is *shown* while
keeping the environment underneath identical.

Two kinds of perturbation:

  schema     changes the tool surface (names, descriptions, argument shapes). The agent
             sees the perturbed schema and its calls are translated back to canonical
             form; a call using pre-perturbation names simply does not resolve.
  transport  changes tool *responses* (dropout, malformed payloads, truncation, latency,
             flakiness) without touching the schema.

Perturbations are deterministic in their seed, so an evaluation is reproducible.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentgym.mcp.tools import REGISTRY, Tool, ToolError


class Kind(str, Enum):
    NONE = "none"
    RENAME_TOOL = "rename_tool"
    PARAPHRASE = "paraphrase"
    RENAME_ARG = "rename_arg"
    ADD_REQUIRED_ARG = "add_required_arg"
    RETYPE_ARG = "retype_arg"
    DISTRACTOR_TOOLS = "distractor_tools"
    # transport
    DROPOUT = "dropout"
    MALFORMED = "malformed"
    TRUNCATE = "truncate"
    FLAKY = "flaky"


SCHEMA_KINDS = (Kind.RENAME_TOOL, Kind.PARAPHRASE, Kind.RENAME_ARG,
                Kind.ADD_REQUIRED_ARG, Kind.RETYPE_ARG, Kind.DISTRACTOR_TOOLS)
TRANSPORT_KINDS = (Kind.DROPOUT, Kind.MALFORMED, Kind.TRUNCATE, Kind.FLAKY)

# Plausible alternative names, so a rename is a realistic API revision rather than noise.
_RENAMES: dict[str, str] = {
    "list_pods": "enumerate_workload_instances",
    "describe_pod": "fetch_instance_detail",
    "get_pod_logs": "fetch_container_output",
    "list_events": "fetch_cluster_events",
    "get_deployment": "fetch_workload_spec",
    "list_deployments": "enumerate_workloads",
    "patch_deployment_resources": "update_workload_resources",
    "scale_deployment": "set_workload_replicas",
    "set_image": "update_workload_image",
    "patch_deployment_probe": "update_health_check",
    "patch_deployment_node_selector": "update_placement_constraints",
    "patch_deployment_secret_ref": "update_credential_binding",
    "rollout_restart": "recycle_workload",
    "delete_pod": "terminate_instance",
    "get_service_health": "fetch_service_status",
    "query_metrics": "run_metric_query",
    "list_metric_names": "enumerate_metrics",
    "get_slo_status": "fetch_slo_report",
    "list_alerts": "fetch_active_alerts",
    "trace_request": "run_synthetic_trace",
    "get_configmap": "fetch_config",
    "list_configmaps": "enumerate_configs",
    "patch_configmap": "update_config_entry",
    "get_secret_meta": "enumerate_credentials",
    "get_service": "fetch_service_spec",
    "patch_service_selector": "update_service_targeting",
    "list_network_policies": "enumerate_network_rules",
    "delete_network_policy": "remove_network_rule",
    "get_resource_quota": "fetch_namespace_limits",
    "patch_resource_quota": "update_namespace_limits",
    "search_runbooks": "query_knowledge_base",
    "get_runbook": "fetch_article",
    "get_service_topology": "fetch_architecture",
    "get_recent_changes": "fetch_deploy_history",
    "submit_diagnosis": "report_root_cause",
}

_ARG_RENAMES: dict[str, str] = {
    "namespace": "ns",
    "name": "resource_name",
    "tail_lines": "max_lines",
    "label_selector": "selector",
    "replicas": "replica_count",
    "image": "image_ref",
    "query": "expr",
    "key": "entry_key",
    "value": "entry_value",
    "service": "service_name",
}

# An argument the perturbed API demands but the environment ignores. A robust agent
# supplies it; a brittle one omits it and gets a validation error it must recover from.
_EXTRA_ARGS: dict[str, dict[str, Any]] = {
    "request_id": {"type": "string", "description": "Caller-supplied idempotency key."},
    "api_version": {"type": "string", "description": "Tool contract version, e.g. \"v2\"."},
}


@dataclass
class PerturbationSpec:
    kinds: tuple[Kind, ...] = ()
    severity: float = 0.5        # fraction of tools affected, or failure probability
    seed: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.kinds) and Kind.NONE not in self.kinds

    def label(self) -> str:
        return "none" if not self.enabled else "+".join(k.value for k in self.kinds)


@dataclass
class PerturbedToolset:
    """The tool surface an agent actually sees, plus translation back to canonical form."""

    spec: PerturbationSpec
    tools: list[Tool] = field(default_factory=list)
    _tool_alias: dict[str, str] = field(default_factory=dict)     # presented -> canonical
    _arg_alias: dict[str, dict[str, str]] = field(default_factory=dict)  # canonical tool -> {presented arg: canonical arg}
    _required_extra: dict[str, list[str]] = field(default_factory=dict)
    _retyped: dict[str, dict[str, str]] = field(default_factory=dict)
    _dropped: set[str] = field(default_factory=set)
    _distractors: set[str] = field(default_factory=set)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools]

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self.tools]

    def resolve(self, presented_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Translate a call against the perturbed surface into a canonical call.

        Raises ToolError for anything the perturbed contract rejects -- an unknown name,
        a missing required argument, a distractor tool, or a dropped tool.
        """
        if presented_name in self._distractors:
            raise ToolError(
                f"{presented_name} is not available in this environment; "
                "it does not operate on the live cluster"
            )
        canonical = self._tool_alias.get(presented_name)
        if canonical is None:
            raise ToolError(
                f"unknown tool {presented_name!r}. Call one of: "
                + ", ".join(sorted(t.name for t in self.tools)[:12]) + " ..."
            )
        if canonical in self._dropped:
            raise ToolError(
                f"{presented_name} is temporarily unavailable (MCP server returned 503)"
            )

        for extra in self._required_extra.get(canonical, []):
            if extra not in args:
                raise ToolError(
                    f"missing required argument {extra!r} for {presented_name}"
                )

        alias = self._arg_alias.get(canonical, {})
        out: dict[str, Any] = {}
        for k, v in args.items():
            if k in _EXTRA_ARGS:
                continue  # accepted, then ignored
            out[alias.get(k, k)] = v

        # Undo any retyping so the implementation receives what it expects.
        for arg, newtype in self._retyped.get(canonical, {}).items():
            canon_arg = alias.get(arg, arg)
            if canon_arg in out and newtype == "string":
                try:
                    out[canon_arg] = int(out[canon_arg])
                except (TypeError, ValueError):
                    raise ToolError(
                        f"argument {arg!r} of {presented_name} must be an integer-valued string"
                    ) from None
        return canonical, out


def build(spec: PerturbationSpec, servers: list[str] | None = None) -> PerturbedToolset:
    """Apply `spec` to the registry and return the surface the agent will see."""
    base = [t for t in REGISTRY.values() if servers is None or t.server in servers]
    ts = PerturbedToolset(spec=spec)

    if not spec.enabled:
        ts.tools = list(base)
        ts._tool_alias = {t.name: t.name for t in base}
        return ts

    rng = random.Random(spec.seed)
    kinds = set(spec.kinds)
    # submit_diagnosis is the terminal action; perturbing it away would make episodes
    # unfinishable for reasons unrelated to the skill being measured.
    perturbable = [t for t in base if t.name != "submit_diagnosis"]
    n_affected = max(1, int(len(perturbable) * spec.severity))
    affected = set(rng.sample([t.name for t in perturbable], k=min(n_affected, len(perturbable))))

    presented: list[Tool] = []
    for t in base:
        name, desc = t.name, t.description
        params = json.loads(json.dumps(t.parameters))  # deep copy
        hit = t.name in affected

        if hit and Kind.RENAME_TOOL in kinds and t.name in _RENAMES:
            name = _RENAMES[t.name]
        if hit and Kind.PARAPHRASE in kinds:
            desc = _paraphrase(desc, rng)

        arg_alias: dict[str, str] = {}
        if hit and Kind.RENAME_ARG in kinds:
            props, required = params["properties"], params.get("required", [])
            newprops, newreq = {}, []
            for arg, schema in props.items():
                new = _ARG_RENAMES.get(arg, arg)
                newprops[new] = schema
                if arg != new:
                    arg_alias[new] = arg
            for arg in required:
                newreq.append(_ARG_RENAMES.get(arg, arg))
            params["properties"], params["required"] = newprops, newreq

        if hit and Kind.ADD_REQUIRED_ARG in kinds:
            extra = rng.choice(list(_EXTRA_ARGS))
            params["properties"][extra] = _EXTRA_ARGS[extra]
            params.setdefault("required", []).append(extra)
            ts._required_extra.setdefault(t.name, []).append(extra)

        if hit and Kind.RETYPE_ARG in kinds:
            for arg, schema in list(params["properties"].items()):
                if schema.get("type") == "integer":
                    params["properties"][arg] = {
                        "type": "string",
                        "description": "Integer encoded as a string, e.g. \"3\".",
                    }
                    ts._retyped.setdefault(t.name, {})[arg] = "string"
                    break

        if hit and Kind.DROPOUT in kinds:
            ts._dropped.add(t.name)

        presented.append(Tool(name, t.server, t.tier, desc, params, t.fn))
        ts._tool_alias[name] = t.name
        if arg_alias:
            ts._arg_alias[t.name] = arg_alias

    if Kind.DISTRACTOR_TOOLS in kinds:
        for real in rng.sample([t for t in presented if t.name in _RENAMES.values()
                                or t.name in REGISTRY],
                               k=min(4, len(presented))):
            fake_name = _distractor_name(real.name, rng)
            if fake_name in ts._tool_alias:
                continue
            ts._distractors.add(fake_name)
            presented.append(Tool(
                fake_name, real.server, real.tier,
                _distractor_desc(real.description, rng), real.parameters, real.fn,
            ))

    ts.tools = presented
    return ts


def _paraphrase(desc: str, rng: random.Random) -> str:
    swaps = [
        ("List", "Enumerate"), ("Read", "Retrieve"), ("Set", "Configure"),
        ("Delete", "Remove"), ("Query", "Search"), ("pods", "workload instances"),
        ("deployment", "workload"), ("namespace", "project"), ("ConfigMap", "config object"),
        ("Service", "endpoint group"), ("container", "process"),
    ]
    out = desc
    for a, b in rng.sample(swaps, k=min(4, len(swaps))):
        out = out.replace(a, b)
    return out


def _distractor_name(real: str, rng: random.Random) -> str:
    suffix = rng.choice(["_v1", "_legacy", "_preview", "_beta"])
    return f"{real}{suffix}"


def _distractor_desc(desc: str, rng: random.Random) -> str:
    return desc + " (Deprecated variant retained for compatibility.)"


# --- transport-level response perturbation ----------------------------------

@dataclass
class TransportPerturber:
    """Corrupts tool *results* without changing the schema."""

    spec: PerturbationSpec
    _rng: random.Random = field(init=False)
    _calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.spec.seed ^ 0x5EED)

    def apply(self, tool_name: str, payload: str) -> str:
        if not self.spec.enabled:
            return payload
        kinds = set(self.spec.kinds)
        self._calls += 1

        if Kind.FLAKY in kinds and self._rng.random() < self.spec.severity * 0.5:
            raise ToolError(
                f"{tool_name}: MCP transport error (connection reset); the call may be retried"
            )
        if Kind.MALFORMED in kinds and self._rng.random() < self.spec.severity * 0.5:
            # A payload that starts as valid JSON and stops mid-object, which is what a
            # truncated streaming response actually looks like.
            return payload[: max(8, len(payload) // 3)] + ' , "…": '
        if Kind.TRUNCATE in kinds and len(payload) > 400:
            keep = max(200, int(len(payload) * (1.0 - self.spec.severity)))
            return payload[:keep] + "\n... [response truncated by MCP gateway]"
        return payload


# --- named suites used by the evaluation matrix ------------------------------

SUITES: dict[str, PerturbationSpec] = {
    "clean": PerturbationSpec(),
    "rename": PerturbationSpec((Kind.RENAME_TOOL,), severity=1.0),
    "paraphrase": PerturbationSpec((Kind.PARAPHRASE,), severity=1.0),
    "arg_rename": PerturbationSpec((Kind.RENAME_ARG,), severity=1.0),
    "extra_required_arg": PerturbationSpec((Kind.ADD_REQUIRED_ARG,), severity=0.6),
    "retype_arg": PerturbationSpec((Kind.RETYPE_ARG,), severity=0.6),
    "distractors": PerturbationSpec((Kind.DISTRACTOR_TOOLS,), severity=0.5),
    "server_dropout": PerturbationSpec((Kind.DROPOUT,), severity=0.25),
    "malformed": PerturbationSpec((Kind.MALFORMED,), severity=0.4),
    "truncation": PerturbationSpec((Kind.TRUNCATE,), severity=0.5),
    "flaky": PerturbationSpec((Kind.FLAKY,), severity=0.3),
    "combined": PerturbationSpec(
        (Kind.RENAME_TOOL, Kind.RENAME_ARG, Kind.PARAPHRASE, Kind.DISTRACTOR_TOOLS),
        severity=0.8,
    ),
}

__all__ = [
    "Kind", "PerturbationSpec", "PerturbedToolset", "TransportPerturber",
    "build", "SUITES", "SCHEMA_KINDS", "TRANSPORT_KINDS",
]
