"""The tool surface the agent acts through.

Every tool declares a JSON Schema, a permission tier, and an implementation bound to a
live `Episode`. The same registry backs both the in-process fast path used for rollout
and the real MCP servers in `agentgym.mcp.servers`, so the tools an agent trains against
are byte-for-byte the tools it is served in production.

Permission tiers:
    READ       observation only, always allowed, safe to call in parallel
    MUTATE     changes cluster state, recorded and reversible
    DESTRUCTIVE  may lose work; gated by policy and (optionally) human approval
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kubernetes.client.rest import ApiException

from agentgym.cluster.kubelet_sim import Verdict
from agentgym.incidents.episode import Episode
from agentgym.incidents.faults import ROOT_CAUSES


class Tier(str, Enum):
    READ = "read"
    MUTATE = "mutate"
    DESTRUCTIVE = "destructive"


class ToolError(RuntimeError):
    """Raised for a call the environment rejects. Surfaced to the agent as an error
    observation rather than crashing the episode -- recovering from these is the skill
    being trained."""


@dataclass(frozen=True)
class Tool:
    name: str
    server: str
    tier: Tier
    description: str
    parameters: dict[str, Any]           # JSON Schema (object)
    fn: Callable[[Episode, dict[str, Any]], Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


_S = {"type": "string"}
_I = {"type": "integer"}


REGISTRY: dict[str, Tool] = {}


def tool(name: str, server: str, tier: Tier, description: str,
         parameters: dict[str, Any]) -> Callable:
    def wrap(fn):
        REGISTRY[name] = Tool(name, server, tier, description, parameters, fn)
        return fn
    return wrap


# =============================================================================
# mcp-kubernetes
# =============================================================================

@tool("list_pods", "mcp-kubernetes", Tier.READ,
      "List pods in a namespace with phase, readiness, restart count and node.",
      _obj({"namespace": _S, "label_selector": _S}, ["namespace"]))
def _list_pods(ep: Episode, a: dict) -> Any:
    sel = a.get("label_selector") or None
    pods = ep.kube.core.list_namespaced_pod(a["namespace"], label_selector=sel).items
    out = []
    for p in pods:
        cs = (p.status.container_statuses or [None])[0]
        verdict = ep.sim.verdict_for(a["namespace"], p.metadata.name)
        out.append({
            "name": p.metadata.name,
            "phase": p.status.phase,
            "reason": _pod_reason(p, verdict),
            "ready": bool(cs and cs.ready),
            "restarts": int(cs.restart_count) if cs else 0,
            "node": p.spec.node_name or None,
            "labels": dict(p.metadata.labels or {}),
        })
    return sorted(out, key=lambda x: x["name"])


@tool("describe_pod", "mcp-kubernetes", Tier.READ,
      "Full detail for one pod: containers, resources, probes, conditions, and status.",
      _obj({"namespace": _S, "name": _S}, ["namespace", "name"]))
def _describe_pod(ep: Episode, a: dict) -> Any:
    try:
        p = ep.kube.core.read_namespaced_pod(name=a["name"], namespace=a["namespace"])
    except ApiException as exc:
        raise ToolError(f"pod {a['name']} not found in {a['namespace']}") from exc
    c = p.spec.containers[0]
    cs = (p.status.container_statuses or [None])[0]
    probe = c.readiness_probe
    return {
        "name": p.metadata.name,
        "namespace": p.metadata.namespace,
        "node": p.spec.node_name,
        "phase": p.status.phase,
        "labels": dict(p.metadata.labels or {}),
        "container": {
            "name": c.name,
            "image": c.image,
            "resources": {
                "limits": dict(c.resources.limits or {}) if c.resources else {},
                "requests": dict(c.resources.requests or {}) if c.resources else {},
            },
            "env_from": [
                {"secret": s.secret_ref.name} if s.secret_ref
                else {"configmap": s.config_map_ref.name}
                for s in (c.env_from or [])
            ],
            "readiness_probe": ({
                "path": probe.http_get.path, "port": probe.http_get.port,
                "initial_delay_seconds": probe.initial_delay_seconds,
                "period_seconds": probe.period_seconds,
                "failure_threshold": probe.failure_threshold,
            } if probe and probe.http_get else None),
        },
        "node_selector": dict(p.spec.node_selector or {}),
        "status": {
            "ready": bool(cs and cs.ready),
            "restarts": int(cs.restart_count) if cs else 0,
            "state": _state_of(cs),
            "last_state": _last_state_of(cs),
        },
        "conditions": [
            {"type": c_.type, "status": c_.status, "reason": c_.reason}
            for c_ in (p.status.conditions or [])
        ],
    }


@tool("get_pod_logs", "mcp-kubernetes", Tier.READ,
      "Read container logs for a pod. Use previous=true to read the logs of the "
      "container instance that crashed.",
      _obj({"namespace": _S, "name": _S, "tail_lines": _I,
            "previous": {"type": "boolean"}}, ["namespace", "name"]))
def _get_pod_logs(ep: Episode, a: dict) -> Any:
    return ep.telemetry.pod_logs(
        a["namespace"], a["name"],
        tail_lines=int(a.get("tail_lines", 50)),
        previous=bool(a.get("previous", False)),
    )


@tool("list_events", "mcp-kubernetes", Tier.READ,
      "Recent Kubernetes events in a namespace, newest last.",
      _obj({"namespace": _S, "limit": _I}, ["namespace"]))
def _list_events(ep: Episode, a: dict) -> Any:
    evs = ep.kube.core.list_namespaced_event(a["namespace"]).items
    out = [{
        "object": f"{e.involved_object.kind}/{e.involved_object.name}",
        "type": e.type, "reason": e.reason, "message": e.message, "count": e.count,
    } for e in evs]
    return out[-int(a.get("limit", 30)):]


@tool("get_deployment", "mcp-kubernetes", Tier.READ,
      "Deployment spec and rollout status: replicas, image, resources, probe, selector.",
      _obj({"namespace": _S, "name": _S}, ["namespace", "name"]))
def _get_deployment(ep: Episode, a: dict) -> Any:
    try:
        d = ep.kube.apps.read_namespaced_deployment(name=a["name"], namespace=a["namespace"])
    except ApiException as exc:
        raise ToolError(f"deployment {a['name']} not found in {a['namespace']}") from exc
    c = d.spec.template.spec.containers[0]
    probe = c.readiness_probe
    return {
        "name": d.metadata.name,
        "replicas_desired": d.spec.replicas,
        "replicas_ready": d.status.ready_replicas or 0,
        "replicas_available": d.status.available_replicas or 0,
        "replicas_updated": d.status.updated_replicas or 0,
        "image": c.image,
        "resources": {
            "limits": dict(c.resources.limits or {}) if c.resources else {},
            "requests": dict(c.resources.requests or {}) if c.resources else {},
        },
        "node_selector": dict(d.spec.template.spec.node_selector or {}),
        "env_from": [
            {"secret": s.secret_ref.name} if s.secret_ref
            else {"configmap": s.config_map_ref.name}
            for s in (c.env_from or [])
        ],
        "readiness_probe": ({"path": probe.http_get.path, "port": probe.http_get.port}
                            if probe and probe.http_get else None),
        "conditions": [
            {"type": c_.type, "status": c_.status, "reason": c_.reason}
            for c_ in (d.status.conditions or [])
        ],
    }


@tool("list_deployments", "mcp-kubernetes", Tier.READ,
      "List deployments in a namespace with desired and ready replica counts.",
      _obj({"namespace": _S}, ["namespace"]))
def _list_deployments(ep: Episode, a: dict) -> Any:
    ds = ep.kube.apps.list_namespaced_deployment(a["namespace"]).items
    return [{
        "name": d.metadata.name,
        "desired": d.spec.replicas,
        "ready": d.status.ready_replicas or 0,
        "image": d.spec.template.spec.containers[0].image,
    } for d in ds]


@tool("patch_deployment_resources", "mcp-kubernetes", Tier.MUTATE,
      "Set a deployment container's memory limit and/or request, in mebibytes. "
      "Use this to give a container more memory than its current limit.",
      _obj({"namespace": _S, "name": _S, "memory_limit_mi": _I,
            "memory_request_mi": _I}, ["namespace", "name"]))
def _patch_resources(ep: Episode, a: dict) -> Any:
    limit, request = a.get("memory_limit_mi"), a.get("memory_request_mi")
    if limit is None and request is None:
        raise ToolError("provide memory_limit_mi and/or memory_request_mi")
    if limit is not None and request is not None and int(request) > int(limit):
        raise ToolError("memory_request_mi cannot exceed memory_limit_mi")
    _require_deployment(ep, a["namespace"], a["name"])
    ep.patch_memory(a["name"],
                    limit_mi=int(limit) if limit is not None else None,
                    request_mi=int(request) if request is not None else None)
    return {"patched": a["name"], "memory_limit_mi": limit, "memory_request_mi": request}


@tool("scale_deployment", "mcp-kubernetes", Tier.MUTATE,
      "Set the replica count of a deployment.",
      _obj({"namespace": _S, "name": _S, "replicas": _I},
           ["namespace", "name", "replicas"]))
def _scale(ep: Episode, a: dict) -> Any:
    n = int(a["replicas"])
    if n < 0:
        raise ToolError("replicas must be >= 0")
    _require_deployment(ep, a["namespace"], a["name"])
    ep.scale(a["name"], n)
    return {"scaled": a["name"], "replicas": n}


@tool("set_image", "mcp-kubernetes", Tier.MUTATE,
      "Set the container image of a deployment, triggering a rolling update.",
      _obj({"namespace": _S, "name": _S, "image": _S}, ["namespace", "name", "image"]))
def _set_image(ep: Episode, a: dict) -> Any:
    _require_deployment(ep, a["namespace"], a["name"])
    ep.set_image(a["name"], a["image"])
    return {"deployment": a["name"], "image": a["image"]}


@tool("patch_deployment_probe", "mcp-kubernetes", Tier.MUTATE,
      "Point a deployment's readiness probe at a different HTTP path and port.",
      _obj({"namespace": _S, "name": _S, "path": _S, "port": _I},
           ["namespace", "name", "path", "port"]))
def _patch_probe(ep: Episode, a: dict) -> Any:
    _require_deployment(ep, a["namespace"], a["name"])
    ep.patch_deployment_probe(a["name"], path=a["path"], port=int(a["port"]))
    return {"deployment": a["name"], "readiness_probe": {"path": a["path"], "port": a["port"]}}


@tool("patch_deployment_node_selector", "mcp-kubernetes", Tier.MUTATE,
      "Replace a deployment's nodeSelector. Pass an empty object to remove it entirely.",
      _obj({"namespace": _S, "name": _S,
            "node_selector": {"type": "object", "additionalProperties": _S}},
           ["namespace", "name", "node_selector"]))
def _patch_node_selector(ep: Episode, a: dict) -> Any:
    _require_deployment(ep, a["namespace"], a["name"])
    ep.patch_node_selector(a["name"], dict(a["node_selector"] or {}))
    return {"deployment": a["name"], "node_selector": a["node_selector"]}


@tool("patch_deployment_secret_ref", "mcp-kubernetes", Tier.MUTATE,
      "Repoint the Secret a deployment loads environment variables from.",
      _obj({"namespace": _S, "name": _S, "secret_name": _S},
           ["namespace", "name", "secret_name"]))
def _patch_secret_ref(ep: Episode, a: dict) -> Any:
    _require_deployment(ep, a["namespace"], a["name"])
    ep.patch_deployment_secret_ref(a["name"], a["secret_name"])
    return {"deployment": a["name"], "secret_ref": a["secret_name"]}


@tool("rollout_restart", "mcp-kubernetes", Tier.MUTATE,
      "Restart a deployment's pods by triggering a fresh rollout.",
      _obj({"namespace": _S, "name": _S}, ["namespace", "name"]))
def _rollout_restart(ep: Episode, a: dict) -> Any:
    _require_deployment(ep, a["namespace"], a["name"])
    ep.rollout_restart(a["name"])
    return {"restarted": a["name"]}


@tool("wait_for_rollout", "mcp-kubernetes", Tier.READ,
      "Wait for a deployment's pods to finish rolling out and become Ready. Call this "
      "after changing a deployment, before checking whether the fix worked -- a change "
      "takes time to take effect.",
      _obj({"namespace": _S, "name": _S}, ["namespace", "name"]))
def _wait_for_rollout(ep: Episode, a: dict) -> Any:
    _require_deployment(ep, a["namespace"], a["name"])
    ticks = ep.settle_world(max_ticks=10)
    d = ep.kube.apps.read_namespaced_deployment(name=a["name"], namespace=a["namespace"])
    ready = d.status.ready_replicas or 0
    desired = d.spec.replicas or 0
    return {
        "deployment": a["name"],
        "ready_replicas": ready,
        "desired_replicas": desired,
        "complete": ready >= desired,
        "waited_ticks": ticks,
    }


@tool("delete_pod", "mcp-kubernetes", Tier.DESTRUCTIVE,
      "Delete a single pod. Its controller will create a replacement. This does not fix "
      "configuration problems -- the replacement inherits the same spec.",
      _obj({"namespace": _S, "name": _S}, ["namespace", "name"]))
def _delete_pod(ep: Episode, a: dict) -> Any:
    try:
        ep.delete_pod(a["name"])
    except ApiException as exc:
        raise ToolError(f"pod {a['name']} not found") from exc
    return {"deleted": a["name"]}


# =============================================================================
# mcp-observability
# =============================================================================

@tool("get_service_health", "mcp-observability", Tier.READ,
      "Health of one service or all services in a namespace: availability, endpoint "
      "counts, error rate and latency.",
      _obj({"namespace": _S, "service": _S}, ["namespace"]))
def _service_health(ep: Episode, a: dict) -> Any:
    health = ep.service_health()
    if name := a.get("service"):
        svc = health.get(name)
        if svc is None:
            raise ToolError(f"service {name} not found in {a['namespace']}")
        return svc.as_dict()
    return [s.as_dict() for s in health.values()]


@tool("query_metrics", "mcp-observability", Tier.READ,
      "Query a metric by name, optionally filtered by labels, e.g. "
      "container_memory_working_set_mib{pod=\"api-0\"}. Supports exact label matches only.",
      _obj({"namespace": _S, "query": _S}, ["namespace", "query"]))
def _query_metrics(ep: Episode, a: dict) -> Any:
    try:
        return ep.telemetry.query(a["namespace"], a["query"])
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


@tool("list_metric_names", "mcp-observability", Tier.READ,
      "List every metric name available in a namespace.",
      _obj({"namespace": _S}, ["namespace"]))
def _list_metric_names(ep: Episode, a: dict) -> Any:
    return sorted({k.split("{")[0] for k in ep.telemetry.metrics(a["namespace"])})


@tool("get_slo_status", "mcp-observability", Tier.READ,
      "SLO compliance for the namespace: which services are burning error budget.",
      _obj({"namespace": _S}, ["namespace"]))
def _slo(ep: Episode, a: dict) -> Any:
    out = []
    for name, svc in ep.service_health().items():
        out.append({
            "service": name,
            "availability_slo": 0.99,
            "observed_availability": round(svc.success_rate, 4),
            "breaching": svc.success_rate < 0.99,
            "latency_slo_p95_ms": 250.0,
            "observed_p95_ms": round(svc.p95_ms, 1),
        })
    return out


@tool("list_alerts", "mcp-observability", Tier.READ,
      "Firing alerts derived from current cluster state.",
      _obj({"namespace": _S}, ["namespace"]))
def _alerts(ep: Episode, a: dict) -> Any:
    alerts = []
    for name, svc in ep.service_health().items():
        if not svc.available:
            alerts.append({"severity": "critical", "alert": "ServiceDown",
                           "service": name, "detail": svc.reason})
        elif svc.reason == "DegradedCapacity":
            alerts.append({"severity": "warning", "alert": "CapacityDegraded",
                           "service": name,
                           "detail": f"{svc.ready_endpoints}/{len(svc.endpoints)} endpoints ready"})
    for pod, verdict in ep.pod_verdicts().items():
        if verdict in (Verdict.OOM_CRASHLOOP, Verdict.KV_CACHE_OOM):
            alerts.append({"severity": "critical", "alert": "PodOOMKilled", "pod": pod,
                           "detail": verdict.value})
        elif verdict is Verdict.IMAGE_PULL_BACKOFF:
            alerts.append({"severity": "critical", "alert": "ImagePullFailure", "pod": pod,
                           "detail": verdict.value})
    return alerts


@tool("trace_request", "mcp-observability", Tier.READ,
      "Trace a synthetic request from the entrypoint through the dependency graph, "
      "showing where it fails.",
      _obj({"namespace": _S, "service": _S}, ["namespace"]))
def _trace(ep: Episode, a: dict) -> Any:
    health = ep.service_health()
    start = a.get("service") or ep.topology.entrypoint
    spans, seen = [], set()

    def walk(name: str, depth: int) -> None:
        if name in seen or depth > 6:
            return
        seen.add(name)
        svc = health.get(name)
        if svc is None:
            spans.append({"service": name, "depth": depth, "status": "unknown"})
            return
        spans.append({
            "service": name, "depth": depth,
            "status": "ok" if svc.available else "error",
            "detail": svc.reason,
            "latency_ms": round(svc.p50_ms, 1) if svc.available else None,
        })
        for dep in svc.depends_on:
            walk(dep, depth + 1)

    walk(start, 0)
    return {"entrypoint": start, "spans": spans}


# =============================================================================
# mcp-config
# =============================================================================

@tool("get_configmap", "mcp-config", Tier.READ,
      "Read a ConfigMap's keys and values.",
      _obj({"namespace": _S, "name": _S}, ["namespace", "name"]))
def _get_cm(ep: Episode, a: dict) -> Any:
    data = ep.configmap_data(a["name"])
    if data is None:
        raise ToolError(f"configmap {a['name']} not found in {a['namespace']}")
    return {"name": a["name"], "data": data}


@tool("list_configmaps", "mcp-config", Tier.READ,
      "List ConfigMap names in a namespace.",
      _obj({"namespace": _S}, ["namespace"]))
def _list_cm(ep: Episode, a: dict) -> Any:
    cms = ep.kube.core.list_namespaced_config_map(a["namespace"]).items
    return [c.metadata.name for c in cms if not c.metadata.name.startswith("kube-")]


@tool("patch_configmap", "mcp-config", Tier.MUTATE,
      "Set one key in a ConfigMap.",
      _obj({"namespace": _S, "name": _S, "key": _S, "value": _S},
           ["namespace", "name", "key", "value"]))
def _patch_cm(ep: Episode, a: dict) -> Any:
    if ep.configmap_data(a["name"]) is None:
        raise ToolError(f"configmap {a['name']} not found in {a['namespace']}")
    ep.patch_configmap(a["name"], a["key"], a["value"])
    return {"configmap": a["name"], "key": a["key"], "value": a["value"]}


@tool("get_secret_meta", "mcp-config", Tier.READ,
      "List which Secrets exist and what keys they hold. Values are never returned.",
      _obj({"namespace": _S}, ["namespace"]))
def _secret_meta(ep: Episode, a: dict) -> Any:
    secs = ep.kube.core.list_namespaced_secret(a["namespace"]).items
    return [{
        "name": s.metadata.name,
        "type": s.type,
        "keys": sorted((s.data or {}).keys()),
    } for s in secs if s.type != "kubernetes.io/service-account-token"]


@tool("get_service", "mcp-config", Tier.READ,
      "Read a Service's selector, ports, and current endpoints.",
      _obj({"namespace": _S, "name": _S}, ["namespace", "name"]))
def _get_service(ep: Episode, a: dict) -> Any:
    sel = ep.service_selector(a["name"])
    if sel is None:
        raise ToolError(f"service {a['name']} not found in {a['namespace']}")
    health = ep.service_health().get(a["name"])
    return {
        "name": a["name"], "selector": sel,
        "endpoints": [{"pod": e.pod, "ready": e.ready} for e in (health.endpoints if health else [])],
    }


@tool("patch_service_selector", "mcp-config", Tier.MUTATE,
      "Replace a Service's label selector so it targets a different set of pods.",
      _obj({"namespace": _S, "name": _S,
            "selector": {"type": "object", "additionalProperties": _S}},
           ["namespace", "name", "selector"]))
def _patch_selector(ep: Episode, a: dict) -> Any:
    if ep.service_selector(a["name"]) is None:
        raise ToolError(f"service {a['name']} not found in {a['namespace']}")
    if not a["selector"]:
        raise ToolError("selector cannot be empty; a Service with no selector has no endpoints")
    ep.patch_service_selector(a["name"], dict(a["selector"]))
    return {"service": a["name"], "selector": a["selector"]}


@tool("list_network_policies", "mcp-config", Tier.READ,
      "List NetworkPolicies with the pods they select and the peers they admit.",
      _obj({"namespace": _S}, ["namespace"]))
def _list_netpol(ep: Episode, a: dict) -> Any:
    out = []
    for np in ep.kube.networking.list_namespaced_network_policy(a["namespace"]).items:
        allowed = []
        for rule in np.spec.ingress or []:
            for peer in rule._from or []:
                if peer.pod_selector is not None:
                    allowed.append(dict(peer.pod_selector.match_labels or {}))
        out.append({
            "name": np.metadata.name,
            "selects": dict((np.spec.pod_selector.match_labels or {})
                            if np.spec.pod_selector else {}),
            "policy_types": list(np.spec.policy_types or []),
            "admits_from": allowed,
        })
    return out


@tool("delete_network_policy", "mcp-config", Tier.MUTATE,
      "Delete a NetworkPolicy, removing the restriction it imposes.",
      _obj({"namespace": _S, "name": _S}, ["namespace", "name"]))
def _del_netpol(ep: Episode, a: dict) -> Any:
    try:
        ep.delete_network_policy(a["name"])
    except ApiException as exc:
        raise ToolError(f"networkpolicy {a['name']} not found") from exc
    return {"deleted": a["name"]}


@tool("get_resource_quota", "mcp-config", Tier.READ,
      "Read namespace ResourceQuotas: hard limits and current usage.",
      _obj({"namespace": _S}, ["namespace"]))
def _get_quota(ep: Episode, a: dict) -> Any:
    out = []
    for rq in ep.kube.core.list_namespaced_resource_quota(a["namespace"]).items:
        out.append({
            "name": rq.metadata.name,
            "hard": {k: str(v) for k, v in (rq.spec.hard or {}).items()},
            "used": {k: str(v) for k, v in ((rq.status.used if rq.status else {}) or {}).items()},
        })
    return out


@tool("patch_resource_quota", "mcp-config", Tier.MUTATE,
      "Raise or lower a ResourceQuota's hard limits, e.g. {\"pods\": \"20\"}.",
      _obj({"namespace": _S, "name": _S,
            "hard": {"type": "object", "additionalProperties": _S}},
           ["namespace", "name", "hard"]))
def _patch_quota(ep: Episode, a: dict) -> Any:
    if ep.resource_quota_hard(a["name"]) is None:
        raise ToolError(f"resourcequota {a['name']} not found in {a['namespace']}")
    ep.patch_resource_quota(a["name"], {k: str(v) for k, v in a["hard"].items()})
    return {"resourcequota": a["name"], "hard": a["hard"]}


# =============================================================================
# mcp-knowledge
# =============================================================================

_RUNBOOKS: dict[str, dict[str, str]] = {
    "rb-oom": {
        "title": "Container OOMKilled / CrashLoopBackOff",
        "body": "Symptom: restart count climbing, lastState.reason=OOMKilled.\n"
                "Check container_memory_working_set_mib against container_memory_limit_mib.\n"
                "Fix: raise the deployment's memory limit above the observed working set with "
                "patch_deployment_resources. Restarting or deleting the pod does not help -- "
                "the replacement inherits the same limit.",
    },
    "rb-image": {
        "title": "ImagePullBackOff",
        "body": "Symptom: pods Pending, events show 'Failed to pull image ... manifest unknown'.\n"
                "Cause is almost always a tag that does not exist. Compare the running image "
                "against the last known-good tag from get_recent_changes, then set_image back.",
    },
    "rb-config": {
        "title": "CrashLoopBackOff after a config change",
        "body": "Symptom: logs show 'required key ... is missing or empty' and the container "
                "exits 1 immediately.\nRead the ConfigMap with get_configmap, find the blank "
                "key, and restore it with patch_configmap.",
    },
    "rb-secret": {
        "title": "CreateContainerConfigError",
        "body": "The pod references a Secret that does not exist. List real secrets with "
                "get_secret_meta and compare against the deployment's env_from. A typo in the "
                "secret name is fixed with patch_deployment_secret_ref.",
    },
    "rb-endpoints": {
        "title": "Service has no endpoints but pods are healthy",
        "body": "If every pod is Ready yet the Service reports zero endpoints, the Service's "
                "selector no longer matches the pod labels. Compare get_service selector with "
                "the pod labels from list_pods and repair with patch_service_selector.",
    },
    "rb-probe": {
        "title": "Pods Running but never Ready",
        "body": "A readiness probe pointing at a path or port the app does not serve keeps "
                "pods out of the Service forever. The app's real health endpoint is in its "
                "logs; realign the probe with patch_deployment_probe.",
    },
    "rb-quota": {
        "title": "ReplicaSet cannot create pods",
        "body": "Events show 'exceeded quota'. Read get_resource_quota; if hard.pods is below "
                "the sum of desired replicas, raise it with patch_resource_quota.",
    },
    "rb-scheduling": {
        "title": "Pods stuck Pending / Unschedulable",
        "body": "describe_pod shows a FailedScheduling event naming the unsatisfied "
                "constraint. A nodeSelector for a tier that does not exist is cleared with "
                "patch_deployment_node_selector.",
    },
    "rb-kvcache": {
        "title": "Inference server fails to allocate KV cache",
        "body": "vLLM logs 'No available memory for the cache blocks'. The KV cache must fit "
                "in the memory left after weights load. Either raise the memory limit "
                "(patch_deployment_resources) or lower MAX_MODEL_LEN in the ConfigMap.",
    },
    "rb-netpol": {
        "title": "Healthy pods that cannot reach a dependency",
        "body": "When every pod is Ready but a consumer reports its dependency unreachable, "
                "check list_network_policies. A policy admitting only a tier that is not "
                "deployed blocks all ingress; delete_network_policy restores connectivity.",
    },
    "rb-node": {
        "title": "Node NotReady",
        "body": "Pods on a node that stopped reporting show phase Unknown and are not "
                "replaced automatically. Capacity silently drops. Restore the node, or "
                "reschedule the workload elsewhere.",
    },
}


@tool("search_runbooks", "mcp-knowledge", Tier.READ,
      "Search the operations runbooks by keyword.",
      _obj({"query": _S}, ["query"]))
def _search_runbooks(ep: Episode, a: dict) -> Any:
    q = a["query"].lower().split()
    scored = []
    for rid, rb in _RUNBOOKS.items():
        blob = (rb["title"] + " " + rb["body"]).lower()
        score = sum(blob.count(term) for term in q)
        if score:
            scored.append((score, rid, rb["title"]))
    scored.sort(reverse=True)
    return [{"id": rid, "title": title} for _, rid, title in scored[:5]]


@tool("get_runbook", "mcp-knowledge", Tier.READ,
      "Read one runbook in full by id.",
      _obj({"id": _S}, ["id"]))
def _get_runbook(ep: Episode, a: dict) -> Any:
    rb = _RUNBOOKS.get(a["id"])
    if rb is None:
        raise ToolError(f"unknown runbook {a['id']}; use search_runbooks first")
    return rb


@tool("get_service_topology", "mcp-knowledge", Tier.READ,
      "The intended architecture: services, their dependencies, and the entrypoint.",
      _obj({"namespace": _S}, ["namespace"]))
def _topology(ep: Episode, a: dict) -> Any:
    return {
        "application": ep.topology.name,
        "description": ep.topology.description,
        "entrypoint": ep.topology.entrypoint,
        "services": [{
            "name": s.name,
            "desired_replicas": s.replicas,
            "depends_on": list(s.depends_on),
            "health_endpoint": f"{s.health_path}:{s.health_port}",
        } for s in ep.topology.services],
    }


@tool("get_recent_changes", "mcp-knowledge", Tier.READ,
      "Deployment history for the namespace: what changed most recently, and by whom. "
      "The cause of an incident is often the newest entry.",
      _obj({"namespace": _S}, ["namespace"]))
def _recent_changes(ep: Episode, a: dict) -> Any:
    return ep.change_log()


@tool("submit_diagnosis", "mcp-knowledge", Tier.READ,
      "Terminal action. Submit the root cause once the incident is fixed. "
      f"root_cause must be one of: {', '.join(ROOT_CAUSES)}",
      _obj({"root_cause": {"type": "string", "enum": list(ROOT_CAUSES)},
            "summary": _S}, ["root_cause", "summary"]))
def _submit(ep: Episode, a: dict) -> Any:
    return {"submitted": True, "root_cause": a["root_cause"], "summary": a["summary"]}


# --- helpers -----------------------------------------------------------------

def _require_deployment(ep: Episode, namespace: str, name: str) -> None:
    if ep._deployment(name) is None:
        raise ToolError(f"deployment {name} not found in {namespace}")


def _pod_reason(pod, verdict: Verdict | None) -> str | None:
    cs = (pod.status.container_statuses or [None])[0]
    if cs is not None:
        if cs.state and cs.state.waiting:
            return cs.state.waiting.reason
        if cs.state and cs.state.terminated:
            return cs.state.terminated.reason
    for cond in pod.status.conditions or []:
        if cond.type == "PodScheduled" and cond.status == "False":
            return cond.reason or "Unschedulable"
    return None


def _state_of(cs) -> dict | None:
    if cs is None or cs.state is None:
        return None
    if cs.state.waiting:
        return {"waiting": {"reason": cs.state.waiting.reason,
                            "message": cs.state.waiting.message}}
    if cs.state.running:
        return {"running": True}
    if cs.state.terminated:
        return {"terminated": {"reason": cs.state.terminated.reason,
                               "exit_code": cs.state.terminated.exit_code}}
    return None


def _last_state_of(cs) -> dict | None:
    if cs is None or cs.last_state is None or cs.last_state.terminated is None:
        return None
    t = cs.last_state.terminated
    return {"terminated": {"reason": t.reason, "exit_code": t.exit_code}}


def tools_for_servers(servers: list[str] | None = None) -> list[Tool]:
    if servers is None:
        return list(REGISTRY.values())
    return [t for t in REGISTRY.values() if t.server in servers]


SERVERS = ("mcp-kubernetes", "mcp-observability", "mcp-config", "mcp-knowledge")

__all__ = ["Tool", "Tier", "ToolError", "REGISTRY", "SERVERS", "tools_for_servers", "json"]
