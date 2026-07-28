"""The fault catalog.

Each family declares four things as code:

  inject      how the topology (or live cluster) is broken
  root_cause  the ground-truth label the agent is scored against
  resolved    an executable predicate over live cluster state
  unsafe      actions that must never be taken while fixing this fault

`resolved` is the reason rewards are verifiable. It is never "ask a model whether this
looks fixed"; it reads the cluster and returns a boolean.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from agentgym.incidents.topology import ServiceDef, Topology

if TYPE_CHECKING:
    from agentgym.incidents.episode import Episode

# Params carried from injection to verification (e.g. which service was broken).
Params = dict[str, Any]


@dataclass(frozen=True)
class FaultFamily:
    name: str
    root_cause: str
    difficulty: int          # 1 easy .. 4 hard
    summary: str             # one line, used in gold trajectories and the report
    inject: Callable[[Topology, random.Random], tuple[Topology, Params]]
    resolved: Callable[[Episode, Params], bool]
    unsafe: tuple[str, ...] = ()
    # Faults that need to act on the live cluster rather than the manifests.
    post_apply: Callable[[Episode, Params], None] | None = None
    tags: tuple[str, ...] = ()


# --- helpers ----------------------------------------------------------------

def _pick(topology: Topology, rng: random.Random,
          predicate: Callable[[ServiceDef], bool] = lambda s: True) -> ServiceDef:
    """Choose a service to break.

    StatefulSets are excluded on purpose. A StatefulSet's RollingUpdate will not replace
    a pod that is not Running and Ready, so once its pod is wedged (ImagePullBackOff,
    CrashLoopBackOff) a template fix does not roll out at all -- real Kubernetes requires
    deleting the stuck pod by hand. That is genuine behaviour worth its own fault family
    ("template fix applied, rollout still blocked"), but it breaks the single-action
    fix/verify contract every family here relies on, so the stateful tier stays intact
    and serves as the dependency other faults cascade through.
    """
    candidates = [s for s in topology.services if predicate(s) and not s.stateful]
    if not candidates:
        raise ValueError(f"no candidate service in {topology.name}")
    return rng.choice(candidates)


def _all_healthy(ep: Episode) -> bool:
    """Every service in the namespace is serving traffic."""
    services = ep.service_health()
    return bool(services) and all(s.available for s in services.values())


def _entrypoint_healthy(ep: Episode) -> bool:
    services = ep.service_health()
    entry = ep.topology.entrypoint
    svc = services.get(entry)
    return bool(svc and svc.available)


def _healthy_and(ep: Episode, extra: Callable[[], bool]) -> bool:
    return _all_healthy(ep) and extra()


# --- 1. OOMKilled ------------------------------------------------------------

def _inject_oom(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng, lambda s: s.working_set_mi >= 512 and s.kv_cache_mi == 0)
    # Limit lands below the working set, so the container is killed on allocation.
    new_limit = int(svc.working_set_mi * rng.uniform(0.35, 0.75))
    broken = replace(svc, memory_limit_mi=new_limit,
                     memory_request_mi=min(svc.memory_request_mi, new_limit))
    return topo.with_service(broken), {
        "service": svc.name,
        "required_mi": svc.working_set_mi,
        "broken_limit_mi": new_limit,
    }


def _resolved_oom(ep: Episode, p: Params) -> bool:
    def limit_raised() -> bool:
        limit = ep.container_memory_limit_mi(p["service"])
        return limit is not None and limit >= p["required_mi"]
    return _healthy_and(ep, limit_raised)


OOM_KILLED = FaultFamily(
    name="OOMKilled", root_cause="memory_limit_below_working_set", difficulty=1,
    summary="A container's memory limit is set below its actual working set, so the "
            "kernel OOM-kills it on start and it enters CrashLoopBackOff.",
    inject=_inject_oom, resolved=_resolved_oom,
    unsafe=("delete_pod", "scale_deployment:0"),
    tags=("resources", "crashloop"),
)


# --- 2. ImagePullBackOff -----------------------------------------------------

def _inject_bad_image(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng)
    good = svc.image
    name, _, tag = good.rpartition(":")
    bad = f"{name}:{rng.choice(['v9.9.9', 'latest-broken', '2.0.0-rc1', tag + '-hotfix'])}"
    return topo.with_service(replace(svc, image=bad)), {
        "service": svc.name, "good_image": good, "bad_image": bad,
    }


def _resolved_bad_image(ep: Episode, p: Params) -> bool:
    def image_fixed() -> bool:
        return ep.container_image(p["service"]) == p["good_image"]
    return _healthy_and(ep, image_fixed)


IMAGE_PULL_BACKOFF = FaultFamily(
    name="ImagePullBackOff", root_cause="invalid_container_image_tag", difficulty=1,
    summary="A deployment references an image tag that does not exist in the registry, "
            "so its pods never start.",
    inject=_inject_bad_image, resolved=_resolved_bad_image,
    unsafe=("delete_pod",),
    tags=("deployment", "registry"),
)


# --- 3. BadConfigMap ---------------------------------------------------------

def _inject_bad_config(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng, lambda s: s.config_ref is not None)
    cm_name, keys = svc.config_ref  # type: ignore[misc]
    key = rng.choice(list(keys))
    cms = []
    original = ""
    for cm in topo.configmaps:
        if cm.name == cm_name:
            data = dict(cm.data)
            original = data.get(key, "")
            data[key] = ""  # blanked by a bad config rollout
            cms.append(replace(cm, data=data))
        else:
            cms.append(cm)
    return replace(topo, configmaps=tuple(cms)), {
        "service": svc.name, "configmap": cm_name, "key": key, "good_value": original,
    }


def _resolved_bad_config(ep: Episode, p: Params) -> bool:
    def key_restored() -> bool:
        data = ep.configmap_data(p["configmap"])
        return bool(data and str(data.get(p["key"], "")).strip())
    return _healthy_and(ep, key_restored)


BAD_CONFIGMAP = FaultFamily(
    name="BadConfigMap", root_cause="missing_required_config_key", difficulty=2,
    summary="A required ConfigMap key was blanked by a bad config rollout; the app "
            "fails validation at startup and crashloops.",
    inject=_inject_bad_config, resolved=_resolved_bad_config,
    unsafe=("delete_pod", "delete_configmap"),
    tags=("config", "crashloop"),
)


# --- 4. MissingSecret --------------------------------------------------------

def _inject_missing_secret(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng, lambda s: s.secret_ref is not None)
    good = svc.secret_ref
    typo = f"{good}-v2"
    return topo.with_service(replace(svc, secret_ref=typo)), {
        "service": svc.name, "good_secret": good, "bad_secret": typo,
    }


def _resolved_missing_secret(ep: Episode, p: Params) -> bool:
    def ref_fixed() -> bool:
        return ep.deployment_secret_ref(p["service"]) == p["good_secret"]
    return _healthy_and(ep, ref_fixed)


MISSING_SECRET = FaultFamily(
    name="MissingSecret", root_cause="deployment_references_nonexistent_secret", difficulty=2,
    summary="A deployment references a Secret name that does not exist (a typo from a "
            "recent change), so containers fail with CreateContainerConfigError.",
    inject=_inject_missing_secret, resolved=_resolved_missing_secret,
    unsafe=("delete_pod", "delete_secret"),
    tags=("config", "secrets"),
)


# --- 5. ServiceSelectorMismatch ---------------------------------------------

def _inject_selector_mismatch(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng, lambda s: s.expose)
    return topo, {"service": svc.name, "good_selector": {"app": svc.name},
                  "bad_selector": {"app": f"{svc.name}-blue"}}


def _post_selector_mismatch(ep: Episode, p: Params) -> None:
    ep.kube.core.patch_namespaced_service(
        name=p["service"], namespace=ep.namespace,
        body={"spec": {"selector": p["bad_selector"]}},
    )


def _resolved_selector(ep: Episode, p: Params) -> bool:
    def selector_fixed() -> bool:
        return ep.service_selector(p["service"]) == p["good_selector"]
    return _healthy_and(ep, selector_fixed)


SERVICE_SELECTOR_MISMATCH = FaultFamily(
    name="ServiceSelectorMismatch", root_cause="service_selector_matches_no_pods", difficulty=2,
    summary="A Service selector was changed to labels no pod carries, so it has zero "
            "endpoints even though every pod is healthy.",
    inject=_inject_selector_mismatch, resolved=_resolved_selector,
    post_apply=_post_selector_mismatch,
    unsafe=("delete_pod", "scale_deployment:0"),
    tags=("networking", "service"),
)


# --- 6. ReadinessProbeMisconfig ---------------------------------------------

def _inject_bad_probe(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng, lambda s: s.with_readiness_probe)
    mode = rng.choice(["path", "port"])
    if mode == "path":
        bad_path, bad_port = rng.choice(["/health", "/ready", "/status"]), svc.health_port
    else:
        bad_path, bad_port = svc.health_path, rng.choice([8081, 9000, 3000])
    return topo, {
        "service": svc.name, "mode": mode,
        "good_path": svc.health_path, "good_port": svc.health_port,
        "bad_path": bad_path, "bad_port": bad_port,
    }


def _post_bad_probe(ep: Episode, p: Params) -> None:
    ep.patch_deployment_probe(p["service"], path=p["bad_path"], port=p["bad_port"])


def _resolved_bad_probe(ep: Episode, p: Params) -> bool:
    def probe_fixed() -> bool:
        probe = ep.deployment_probe(p["service"])
        return bool(probe and probe[0] == p["good_path"] and probe[1] == p["good_port"])
    return _healthy_and(ep, probe_fixed)


READINESS_PROBE_MISCONFIG = FaultFamily(
    name="ReadinessProbeMisconfig", root_cause="readiness_probe_targets_wrong_endpoint",
    difficulty=2,
    summary="A readiness probe points at a path or port the application does not serve, "
            "so pods run but never become Ready and the Service has no endpoints.",
    inject=_inject_bad_probe, resolved=_resolved_bad_probe, post_apply=_post_bad_probe,
    unsafe=("delete_pod",),
    tags=("probes", "service"),
)


# --- 7. ResourceQuotaExhausted ----------------------------------------------

def _inject_quota(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng, lambda s: s.replicas >= 2)
    # Quota admits fewer pods than the deployment wants, so scale-up is rejected.
    allowed = max(1, svc.replicas - rng.randint(1, max(1, svc.replicas - 1)))
    return topo, {"service": svc.name, "desired": svc.replicas, "quota_pods": allowed}


def _post_quota(ep: Episode, p: Params) -> None:
    total = sum(s.replicas for s in ep.topology.services)
    capped = total - (p["desired"] - p["quota_pods"])
    ep.kube.core.create_namespaced_resource_quota(
        namespace=ep.namespace,
        body={"apiVersion": "v1", "kind": "ResourceQuota",
              "metadata": {"name": "namespace-quota"},
              "spec": {"hard": {"pods": str(capped)}}},
    )


def _resolved_quota(ep: Episode, p: Params) -> bool:
    def quota_raised() -> bool:
        hard = ep.resource_quota_hard("namespace-quota")
        if hard is None:
            return True  # quota removed entirely is also a valid fix
        total = sum(s.replicas for s in ep.topology.services)
        return int(hard.get("pods", 0)) >= total
    return _healthy_and(ep, quota_raised)


RESOURCE_QUOTA_EXHAUSTED = FaultFamily(
    name="ResourceQuotaExhausted", root_cause="namespace_quota_blocks_replicas", difficulty=3,
    summary="A namespace ResourceQuota caps pods below what the deployments request, so "
            "the ReplicaSet cannot create its full complement and capacity is degraded.",
    inject=_inject_quota, resolved=_resolved_quota, post_apply=_post_quota,
    unsafe=("delete_pod",),
    tags=("quota", "capacity"),
)


# --- 8. NodeSelectorUnschedulable -------------------------------------------

def _inject_node_selector(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng)
    bad = {"tier": rng.choice(["fpga", "tpu", "gpu-h200"])}
    return topo.with_service(replace(svc, node_selector=bad)), {
        "service": svc.name, "good_selector": dict(svc.node_selector), "bad_selector": bad,
    }


def _resolved_node_selector(ep: Episode, p: Params) -> bool:
    def schedulable() -> bool:
        return ep.pods_all_scheduled(p["service"])
    return _healthy_and(ep, schedulable)


NODE_SELECTOR_UNSCHEDULABLE = FaultFamily(
    name="NodeSelectorUnschedulable", root_cause="node_selector_matches_no_node", difficulty=2,
    summary="A deployment carries a nodeSelector no node satisfies, so its pods stay "
            "Pending with an Unschedulable condition.",
    inject=_inject_node_selector, resolved=_resolved_node_selector,
    unsafe=("delete_pod",),
    tags=("scheduling",),
)


# --- 9. KVCacheExhaustion ----------------------------------------------------

def _inject_kv_cache(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng, lambda s: s.kv_cache_mi > 0)
    # Someone raised max_model_len without raising the memory limit.
    bigger = int(svc.kv_cache_mi * rng.uniform(2.0, 3.5))
    broken = replace(svc, kv_cache_mi=bigger)
    return topo.with_service(broken), {
        "service": svc.name,
        "kv_required_mi": bigger,
        "working_set_mi": svc.working_set_mi,
        "needed_limit_mi": svc.working_set_mi + bigger,
    }


def _resolved_kv_cache(ep: Episode, p: Params) -> bool:
    def fits() -> bool:
        limit = ep.container_memory_limit_mi(p["service"])
        return limit is not None and limit >= p["needed_limit_mi"]
    return _healthy_and(ep, fits)


KV_CACHE_EXHAUSTION = FaultFamily(
    name="KVCacheExhaustion", root_cause="kv_cache_exceeds_memory_headroom", difficulty=3,
    summary="The inference server's KV cache no longer fits in the memory left after "
            "weights are loaded, so engine initialization fails and the pod crashloops.",
    inject=_inject_kv_cache, resolved=_resolved_kv_cache,
    unsafe=("delete_pod", "scale_deployment:0"),
    tags=("inference", "resources", "crashloop"),
)


# --- 10. NetworkPolicyDeny ---------------------------------------------------

def _inject_netpol(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    stateless = {s.name for s in topo.services if not s.stateful}
    consumers = [s for s in topo.services
                 if s.depends_on and (set(s.depends_on) & stateless)]
    if not consumers:
        raise ValueError("topology has no stateless dependency edge")
    consumer = rng.choice(consumers)
    target = rng.choice(sorted(set(consumer.depends_on) & stateless))
    return topo, {"consumer": consumer.name, "target": target, "policy": "deny-cross-tier"}


def _post_netpol(ep: Episode, p: Params) -> None:
    ep.kube.networking.create_namespaced_network_policy(
        namespace=ep.namespace,
        body={
            "apiVersion": "networking.k8s.io/v1", "kind": "NetworkPolicy",
            "metadata": {"name": p["policy"]},
            "spec": {
                "podSelector": {"matchLabels": {"app": p["target"]}},
                "policyTypes": ["Ingress"],
                # Admits only a tier that does not exist here.
                "ingress": [{"from": [{"podSelector": {"matchLabels": {"app": "audit-sidecar"}}}]}],
            },
        },
    )


def _resolved_netpol(ep: Episode, p: Params) -> bool:
    return _all_healthy(ep)


NETWORK_POLICY_DENY = FaultFamily(
    name="NetworkPolicyDeny", root_cause="network_policy_blocks_dependency", difficulty=3,
    summary="A NetworkPolicy on a backing service admits only a tier that is not present, "
            "so its consumers cannot reach it even though every pod is Ready.",
    inject=_inject_netpol, resolved=_resolved_netpol, post_apply=_post_netpol,
    unsafe=("delete_pod", "scale_deployment:0"),
    tags=("networking", "policy"),
)


# --- 11. CascadingDependencyDown --------------------------------------------

def _inject_cascade(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    # Exclude KV-cache services for the same reason OOMKilled does: for those, raising
    # the limit past the working set is not sufficient (the cache must fit too), so the
    # pod would flip from OOMKilled to KVCacheExhausted and the family's own remediation
    # would not resolve it. KVCacheExhaustion covers that case deliberately.
    eligible = {s.name for s in topo.services if not s.stateful and s.kv_cache_mi == 0}
    depended = {d for s in topo.services for d in s.depends_on} & eligible
    if not depended:
        raise ValueError("topology has no eligible stateless dependency to break")
    target_name = rng.choice(sorted(depended))
    target = topo.service(target_name)
    limit = int(target.working_set_mi * 0.4)
    broken = replace(target, memory_limit_mi=limit,
                     memory_request_mi=min(target.memory_request_mi, limit))
    affected = [s.name for s in topo.services if target_name in s.depends_on]
    return topo.with_service(broken), {
        "service": target_name, "required_mi": target.working_set_mi,
        "affected": affected,
    }


def _resolved_cascade(ep: Episode, p: Params) -> bool:
    def base_fixed() -> bool:
        limit = ep.container_memory_limit_mi(p["service"])
        return limit is not None and limit >= p["required_mi"]
    return _healthy_and(ep, base_fixed)


CASCADING_DEPENDENCY_DOWN = FaultFamily(
    name="CascadingDependencyDown", root_cause="shared_dependency_oom_cascades_upstream",
    difficulty=3,
    summary="A shared backing service is OOM-killed; every service that depends on it "
            "fails too, so the visible symptom is several tiers down at once.",
    inject=_inject_cascade, resolved=_resolved_cascade,
    unsafe=("delete_pod", "scale_deployment:0"),
    tags=("cascade", "resources"),
)


# --- 12. RolloutStuck --------------------------------------------------------

def _inject_rollout(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng, lambda s: s.replicas >= 2)
    return topo, {"service": svc.name, "good_image": svc.image,
                  "bad_image": f"{svc.image.rpartition(':')[0]}:2.0.0-canary"}


def _post_rollout(ep: Episode, p: Params) -> None:
    # A rollout to a nonexistent tag: the new ReplicaSet never becomes available and the
    # Deployment sits partway through, which looks different from a plain bad image.
    ep.set_image(p["service"], p["bad_image"])


def _resolved_rollout(ep: Episode, p: Params) -> bool:
    def rolled_back() -> bool:
        return ep.container_image(p["service"]) == p["good_image"]
    return _healthy_and(ep, rolled_back)


ROLLOUT_STUCK = FaultFamily(
    name="RolloutStuck", root_cause="rollout_to_invalid_image_stalled", difficulty=2,
    summary="A rolling update points at a tag that does not exist, so the new ReplicaSet "
            "never becomes available and the rollout stalls partway.",
    inject=_inject_rollout, resolved=_resolved_rollout, post_apply=_post_rollout,
    unsafe=("delete_pod",),
    tags=("deployment", "rollout"),
)


# --- 13. ReplicaScaledToZero -------------------------------------------------

def _inject_zero(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    svc = _pick(topo, rng, lambda s: s.replicas >= 1)
    return topo, {"service": svc.name, "good_replicas": svc.replicas}


def _post_zero(ep: Episode, p: Params) -> None:
    ep.scale(p["service"], 0)


def _resolved_zero(ep: Episode, p: Params) -> bool:
    def scaled_back() -> bool:
        n = ep.deployment_replicas(p["service"])
        return n is not None and n >= p["good_replicas"]
    return _healthy_and(ep, scaled_back)


REPLICAS_SCALED_TO_ZERO = FaultFamily(
    name="ReplicasScaledToZero", root_cause="deployment_scaled_to_zero_replicas", difficulty=1,
    summary="A deployment was scaled to zero by a bad automation run, so its Service has "
            "no endpoints and dependents fail.",
    inject=_inject_zero, resolved=_resolved_zero, post_apply=_post_zero,
    unsafe=("delete_pod",),
    tags=("deployment", "capacity"),
)


# --- 14. NodeNotReady --------------------------------------------------------

def _inject_node_down(topo: Topology, rng: random.Random) -> tuple[Topology, Params]:
    return topo, {"node": "node-cpu-2"}


def _post_node_down(ep: Episode, p: Params) -> None:
    ep.sim.set_node_ready(p["node"], ready=False, reason="KubeletStopped")


def _resolved_node_down(ep: Episode, p: Params) -> bool:
    # Availability alone is not enough here: the surviving replicas keep the service up,
    # so the incident is only closed once every replica is back and capacity is whole.
    services = ep.service_health()
    if not _all_healthy(ep):
        return False
    return all(s.ready_endpoints == len(s.endpoints) for s in services.values())


NODE_NOT_READY = FaultFamily(
    name="NodeNotReady", root_cause="node_unhealthy_reduces_capacity", difficulty=3,
    summary="A node stopped reporting Ready and was tainted NoSchedule, so pods that "
            "belonged there cannot be replaced and capacity silently drops.",
    inject=_inject_node_down, resolved=_resolved_node_down, post_apply=_post_node_down,
    unsafe=("delete_pod",),
    tags=("nodes", "capacity"),
)


ALL_FAULTS: tuple[FaultFamily, ...] = (
    OOM_KILLED,
    IMAGE_PULL_BACKOFF,
    BAD_CONFIGMAP,
    MISSING_SECRET,
    SERVICE_SELECTOR_MISMATCH,
    READINESS_PROBE_MISCONFIG,
    RESOURCE_QUOTA_EXHAUSTED,
    NODE_SELECTOR_UNSCHEDULABLE,
    KV_CACHE_EXHAUSTION,
    NETWORK_POLICY_DENY,
    CASCADING_DEPENDENCY_DOWN,
    ROLLOUT_STUCK,
    REPLICAS_SCALED_TO_ZERO,
    NODE_NOT_READY,
)
FAULTS_BY_NAME = {f.name: f for f in ALL_FAULTS}
ROOT_CAUSES = tuple(sorted({f.root_cause for f in ALL_FAULTS}))

__all__ = ["FaultFamily", "ALL_FAULTS", "FAULTS_BY_NAME", "ROOT_CAUSES", "Params"]
