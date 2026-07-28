"""Service topology, endpoints, and a deterministic traffic model.

Kubernetes gives us Services and label selectors; it does not tell us whether the
application behind them actually works. This module closes that gap: it derives
Endpoints the way kube-proxy would, then propagates availability through the dependency
graph so "is the checkout service healthy" is a computed fact rather than a guess.

Everything here is a pure function of cluster state, which is what lets the verifier
assert resolution exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentgym.cluster.client import KubeClient
from agentgym.cluster.kubelet_sim import KubeletSim, Verdict

# Latency floor per service, in milliseconds, before load effects.
BASE_LATENCY_MS = 25.0
# Each additional unit of load over capacity multiplies latency by this much.
OVERLOAD_LATENCY_FACTOR = 3.2


@dataclass
class Endpoint:
    pod: str
    ip: str
    ready: bool


@dataclass
class ServiceHealth:
    name: str
    namespace: str
    selector: dict[str, str]
    endpoints: list[Endpoint] = field(default_factory=list)
    # Populated by `Mesh.evaluate`
    available: bool = False
    success_rate: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    error_rate: float = 1.0
    reason: str = "NoEndpoints"
    depends_on: tuple[str, ...] = ()
    blocked_by_policy: tuple[str, ...] = ()

    @property
    def ready_endpoints(self) -> int:
        return sum(1 for e in self.endpoints if e.ready)

    def as_dict(self) -> dict:
        return {
            "service": self.name,
            "namespace": self.namespace,
            "available": self.available,
            "ready_endpoints": self.ready_endpoints,
            "total_endpoints": len(self.endpoints),
            "success_rate": round(self.success_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "latency_p50_ms": round(self.p50_ms, 1),
            "latency_p95_ms": round(self.p95_ms, 1),
            "reason": self.reason,
            "depends_on": list(self.depends_on),
            "blocked_by_network_policy": list(self.blocked_by_policy),
        }


@dataclass
class Mesh:
    """Computes service-level health for one namespace."""

    kube: KubeClient
    sim: KubeletSim

    def evaluate(self, namespace: str) -> dict[str, ServiceHealth]:
        services = self._collect_services(namespace)
        policies = self._collect_policies(namespace)

        # Resolve availability by relaxation: a service is available if it has ready
        # endpoints and every dependency it needs is reachable and available. Iterating
        # `len(services)` times is enough for any acyclic graph, and cycles settle to
        # "unavailable", which is the honest answer for a deadlocked dependency loop.
        for _ in range(max(1, len(services))):
            changed = False
            for svc in services.values():
                before = (svc.available, svc.reason)
                self._score(svc, services, policies)
                if before != (svc.available, svc.reason):
                    changed = True
            if not changed:
                break
        return services

    # --- collection ---------------------------------------------------------

    def _collect_services(self, namespace: str) -> dict[str, ServiceHealth]:
        pods = self.kube.core.list_namespaced_pod(namespace).items
        out: dict[str, ServiceHealth] = {}

        for svc in self.kube.core.list_namespaced_service(namespace).items:
            selector = dict(svc.spec.selector or {})
            health = ServiceHealth(
                name=svc.metadata.name, namespace=namespace, selector=selector
            )
            if selector:
                for pod in pods:
                    labels = dict(pod.metadata.labels or {})
                    if all(labels.get(k) == v for k, v in selector.items()):
                        verdict = self.sim.verdict_for(namespace, pod.metadata.name)
                        health.endpoints.append(Endpoint(
                            pod=pod.metadata.name,
                            ip=pod.status.pod_ip or "",
                            ready=verdict is Verdict.RUNNING_READY,
                        ))
                        deps = _deps_from_pod(pod)
                        if deps:
                            health.depends_on = deps
            out[svc.metadata.name] = health
        return out

    def _collect_policies(self, namespace: str) -> list[tuple[dict[str, str], list[dict[str, str]]]]:
        """Return (target selector, allowed-from selectors) for each ingress policy."""
        policies = []
        for np in self.kube.networking.list_namespaced_network_policy(namespace).items:
            target = dict((np.spec.pod_selector.match_labels or {}) if np.spec.pod_selector else {})
            if "Ingress" not in (np.spec.policy_types or ["Ingress"]):
                continue
            allowed: list[dict[str, str]] = []
            for rule in np.spec.ingress or []:
                for peer in rule._from or []:
                    if peer.pod_selector is not None:
                        allowed.append(dict(peer.pod_selector.match_labels or {}))
            policies.append((target, allowed))
        return policies

    # --- scoring ------------------------------------------------------------

    def _score(
        self,
        svc: ServiceHealth,
        services: dict[str, ServiceHealth],
        policies: list[tuple[dict[str, str], list[dict[str, str]]]],
    ) -> None:
        if not svc.selector:
            svc.available, svc.reason = False, "NoSelector"
            svc.success_rate, svc.error_rate = 0.0, 1.0
            return
        if not svc.endpoints:
            # A selector that matches nothing is the classic "service points at the
            # wrong labels" outage: pods are healthy, the Service is a black hole.
            svc.available, svc.reason = False, "SelectorMatchesNoPods"
            svc.success_rate, svc.error_rate = 0.0, 1.0
            return
        ready = svc.ready_endpoints
        if ready == 0:
            svc.available, svc.reason = False, "AllEndpointsUnready"
            svc.success_rate, svc.error_rate = 0.0, 1.0
            return

        blocked = []
        for dep_name in svc.depends_on:
            dep = services.get(dep_name)
            if dep is None:
                svc.available, svc.reason = False, f"DependencyMissing:{dep_name}"
                svc.success_rate, svc.error_rate = 0.0, 1.0
                return
            if self._policy_blocks(svc, dep, policies):
                blocked.append(dep_name)
            elif not dep.available:
                svc.available, svc.reason = False, f"DependencyUnavailable:{dep_name}"
                svc.success_rate, svc.error_rate = 0.0, 1.0
                return
        if blocked:
            svc.blocked_by_policy = tuple(blocked)
            svc.available, svc.reason = False, f"NetworkPolicyDenied:{','.join(blocked)}"
            svc.success_rate, svc.error_rate = 0.0, 1.0
            return

        # Healthy, but latency degrades as replicas drop below the declared fleet.
        total = max(1, len(svc.endpoints))
        load = total / ready  # 1.0 when everything is up
        svc.available = True
        svc.reason = "Healthy" if ready == total else "DegradedCapacity"
        svc.success_rate = 1.0
        svc.error_rate = 0.0
        svc.p50_ms = BASE_LATENCY_MS * (load ** 1.5)
        svc.p95_ms = svc.p50_ms * (1.0 + OVERLOAD_LATENCY_FACTOR * (load - 1.0) + 0.8)

    def _policy_blocks(
        self,
        source: ServiceHealth,
        target: ServiceHealth,
        policies: list[tuple[dict[str, str], list[dict[str, str]]]],
    ) -> bool:
        """True when an ingress policy on `target` does not admit `source`."""
        source_labels = source.selector
        for target_sel, allowed in policies:
            if not target_sel or not _matches(target.selector, target_sel):
                continue
            # A policy selecting the target restricts ingress to its allowed peers.
            if not any(_matches(source_labels, peer) for peer in allowed):
                return True
        return False


def _matches(labels: dict[str, str], selector: dict[str, str]) -> bool:
    return bool(selector) and all(labels.get(k) == v for k, v in selector.items())


def _deps_from_pod(pod) -> tuple[str, ...]:
    ann = (pod.metadata.annotations or {})
    raw = ann.get("agentgym.io/depends-on", "")
    return tuple(d for d in raw.split(",") if d)


__all__ = ["Mesh", "ServiceHealth", "Endpoint"]
