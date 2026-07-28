"""One incident episode: a namespace, a broken topology, and the means to check it.

An Episode owns a namespace inside a shared control plane. Creating and tearing down a
namespace is roughly a second, versus ~13s for a whole control plane, so a worker starts
one control plane and runs many episodes through it.

The accessor methods here are the vocabulary the fault predicates in `faults.py` are
written against, and the mutation methods are what the MCP tools ultimately call.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import yaml
from kubernetes.client.rest import ApiException

from agentgym.cluster.client import KubeClient, mib
from agentgym.cluster.controlplane import ControlPlane
from agentgym.cluster.kubelet_sim import KubeletSim, Verdict
from agentgym.cluster.mesh import Mesh, ServiceHealth
from agentgym.cluster.telemetry import Telemetry
from agentgym.incidents.faults import FaultFamily, Params
from agentgym.incidents.topology import Topology

# How many simulation ticks one agent action advances the world.
TICKS_PER_STEP = 2
# How long to let the real controllers (deployment -> RS -> pod -> scheduler) settle.
SETTLE_TIMEOUT_S = 20.0


@dataclass
class Episode:
    """A single incident instance, live in its own namespace."""

    control_plane: ControlPlane
    kube: KubeClient
    sim: KubeletSim
    namespace: str
    topology: Topology
    fault: FaultFamily
    params: Params
    seed: int = 0

    mesh: Mesh = field(init=False)
    telemetry: Telemetry = field(init=False)
    _services_cache: dict[str, ServiceHealth] | None = field(default=None, init=False)
    _actions: list[dict[str, Any]] = field(default_factory=list, init=False)
    # Times the control plane failed to produce its intended pods within SETTLE_TIMEOUT_S.
    # Non-zero means this episode's verdict was computed against a world that had not
    # finished converging, so the reward describes the machine's load as much as the agent.
    settle_timeouts: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.mesh = Mesh(kube=self.kube, sim=self.sim)
        self.telemetry = Telemetry(kube=self.kube, sim=self.sim, mesh=self.mesh, seed=self.seed)

    # --- lifecycle ----------------------------------------------------------

    def setup(self) -> Episode:
        """Create the namespace, apply the broken topology, run post-apply injection."""
        self.kube.core.create_namespace(
            body={"apiVersion": "v1", "kind": "Namespace",
                  "metadata": {"name": self.namespace,
                               "labels": {"agentgym.io/episode": "true"}}}
        )
        if self.namespace not in self.sim.namespaces:
            self.sim.namespaces.append(self.namespace)

        manifests = self.topology.render()
        self.control_plane.kubectl(
            "apply", "-n", self.namespace, "-f", "-",
            stdin=yaml.safe_dump_all(manifests),
        )
        self._settle()
        if self.fault.post_apply is not None:
            self.fault.post_apply(self, self.params)
        self.step(ticks=3)
        return self

    def teardown(self) -> None:
        # Reap before releasing the namespace: a namespace cannot finish deleting while
        # pods remain, and those pods would otherwise keep holding node capacity for the
        # rest of the worker's life.
        try:
            self.kube.core.delete_namespace(name=self.namespace)
        except ApiException:
            pass
        self.sim.reap_namespace(self.namespace)
        self.sim.forget(self.namespace)
        if self.namespace in self.sim.namespaces:
            self.sim.namespaces.remove(self.namespace)

    def step(self, ticks: int = TICKS_PER_STEP) -> None:
        """Advance the world after an agent action."""
        self._settle()
        self.sim.reconcile(ticks=ticks)
        self.telemetry.emit_events(self.namespace)
        self._services_cache = None

    def settle_world(self, max_ticks: int = 14) -> int:
        """Advance until pod state stops changing, or `max_ticks` elapse.

        A rolling update needs several ticks to create, schedule and boot replacement
        pods. Without this, a correct fix looks broken purely because it was checked too
        early -- which would make the reward a measure of polling luck rather than of
        whether the incident was actually resolved.
        """
        previous: dict[str, Any] | None = None
        for tick in range(max_ticks):
            self.step(ticks=1)
            current = {k: v.value for k, v in self.pod_verdicts().items()}
            # "Unchanged" is not "converged". A pod with boot_ticks=2 reports Booting on
            # two consecutive ticks while still making progress, so stopping on equality
            # alone cuts the world off mid-startup and reports a correct fix as failed.
            # Only settle once nothing is still in a transient state.
            transient = any(v == Verdict.BOOTING.value for v in current.values())
            if current and current == previous and not transient:
                return tick + 1
            previous = current
        return max_ticks

    def _settle(self) -> bool:
        """Wait until the real controllers have produced the pods they intend to.

        Polls on a condition rather than sleeping a fixed amount, so the wait is as short
        as the control plane allows and the outcome does not depend on machine speed.

        Returns whether it actually settled. **The timeout is a fail-open**: when it
        expires the world advances with pods the controller-manager has not created yet,
        the incident then looks unresolved, and nothing anywhere reports that the cause was
        a slow control plane rather than a bad fix. That is indistinguishable from a real
        failure at the point where the reward is computed, so every expiry is counted and
        the count travels with the episode.
        """
        deadline = time.time() + SETTLE_TIMEOUT_S
        while time.time() < deadline:
            try:
                deployments = list(
                    self.kube.apps.list_namespaced_deployment(self.namespace).items
                ) + list(
                    self.kube.apps.list_namespaced_stateful_set(self.namespace).items
                )
                pods = self.kube.core.list_namespaced_pod(self.namespace).items
            except ApiException:
                time.sleep(0.1)
                continue
            live = [p for p in pods if p.metadata.deletion_timestamp is None]
            # Every workload controller has created its pods, and the scheduler has made
            # a decision about each one (bound, or explicitly unschedulable).
            want = sum(d.spec.replicas or 0 for d in deployments)
            observed = all(
                (d.status.observed_generation or 0) >= (d.metadata.generation or 0)
                for d in deployments
            )
            decided = all(
                p.spec.node_name or _unschedulable(p) for p in live
            )
            if observed and len(live) >= want and decided:
                return True
            time.sleep(0.15)
        self.settle_timeouts += 1
        return False

    # --- observation --------------------------------------------------------

    def service_health(self) -> dict[str, ServiceHealth]:
        if self._services_cache is None:
            self._services_cache = self.mesh.evaluate(self.namespace)
        return self._services_cache

    def resolved(self) -> bool:
        """The executable verdict: is this incident actually fixed?"""
        try:
            return bool(self.fault.resolved(self, self.params))
        except Exception:
            return False

    def pod_verdicts(self) -> dict[str, Verdict]:
        prefix = f"{self.namespace}/"
        return {k[len(prefix):]: v for k, v in self.sim._verdicts.items()
                if k.startswith(prefix)}

    # --- accessors used by fault predicates ---------------------------------

    def workload_kind(self, name: str) -> str | None:
        """"Deployment" or "StatefulSet" -- whichever owns this service, if either.

        Datastores are StatefulSets and stateless tiers are Deployments, so every
        accessor and every mutation has to dispatch on kind. A tool that only understands
        Deployments silently fails on exactly the services an incident most often
        involves.
        """
        try:
            self.kube.apps.read_namespaced_deployment(name=name, namespace=self.namespace)
            return "Deployment"
        except ApiException:
            pass
        try:
            self.kube.apps.read_namespaced_stateful_set(name=name, namespace=self.namespace)
            return "StatefulSet"
        except ApiException:
            return None

    def _deployment(self, name: str):
        """The workload object for `name`, whichever kind it is."""
        try:
            return self.kube.apps.read_namespaced_deployment(name=name, namespace=self.namespace)
        except ApiException:
            pass
        try:
            return self.kube.apps.read_namespaced_stateful_set(
                name=name, namespace=self.namespace)
        except ApiException:
            return None

    def _patch_workload(self, name: str, body: dict[str, Any]) -> None:
        kind = self.workload_kind(name)
        if kind == "StatefulSet":
            self.kube.apps.patch_namespaced_stateful_set(
                name=name, namespace=self.namespace, body=body)
        elif kind == "Deployment":
            self.kube.apps.patch_namespaced_deployment(
                name=name, namespace=self.namespace, body=body)
        else:
            raise KeyError(f"no Deployment or StatefulSet named {name}")

    def container_memory_limit_mi(self, service: str) -> float | None:
        dep = self._deployment(service)
        if dep is None:
            return None
        c = dep.spec.template.spec.containers[0]
        if c.resources is None or not c.resources.limits:
            return None
        return mib(c.resources.limits.get("memory"))

    def container_image(self, service: str) -> str | None:
        dep = self._deployment(service)
        return dep.spec.template.spec.containers[0].image if dep else None

    def deployment_replicas(self, service: str) -> int | None:
        dep = self._deployment(service)
        return dep.spec.replicas if dep else None

    def deployment_secret_ref(self, service: str) -> str | None:
        dep = self._deployment(service)
        if dep is None:
            return None
        for src in dep.spec.template.spec.containers[0].env_from or []:
            if src.secret_ref is not None:
                return src.secret_ref.name
        return None

    def deployment_probe(self, service: str) -> tuple[str, int] | None:
        dep = self._deployment(service)
        if dep is None:
            return None
        probe = dep.spec.template.spec.containers[0].readiness_probe
        if probe is None or probe.http_get is None:
            return None
        port = probe.http_get.port
        return (probe.http_get.path, int(port) if str(port).isdigit() else -1)

    def configmap_data(self, name: str) -> dict[str, str] | None:
        try:
            cm = self.kube.core.read_namespaced_config_map(name=name, namespace=self.namespace)
            return dict(cm.data or {})
        except ApiException:
            return None

    def service_selector(self, name: str) -> dict[str, str] | None:
        try:
            svc = self.kube.core.read_namespaced_service(name=name, namespace=self.namespace)
            return dict(svc.spec.selector or {})
        except ApiException:
            return None

    def resource_quota_hard(self, name: str) -> dict[str, str] | None:
        try:
            rq = self.kube.core.read_namespaced_resource_quota(name=name, namespace=self.namespace)
            return {k: str(v) for k, v in (rq.spec.hard or {}).items()}
        except ApiException:
            return None

    def pods_all_scheduled(self, service: str) -> bool:
        pods = self.kube.core.list_namespaced_pod(
            self.namespace, label_selector=f"app={service}"
        ).items
        live = [p for p in pods if p.metadata.deletion_timestamp is None]
        return bool(live) and all(p.spec.node_name for p in live)

    # --- mutations (what the MCP write-tools call) --------------------------

    def set_image(self, service: str, image: str) -> None:
        self._patch_workload(service, {"spec": {"template": {"spec": {"containers": [
            {"name": service, "image": image}]}}}})

    def scale(self, service: str, replicas: int) -> None:
        self._patch_workload(service, {"spec": {"replicas": replicas}})

    def patch_deployment_probe(self, service: str, *, path: str, port: int) -> None:
        self._patch_workload(service, {"spec": {"template": {"spec": {"containers": [{
            "name": service,
            "readinessProbe": {
                "httpGet": {"path": path, "port": port},
                "initialDelaySeconds": 5, "periodSeconds": 10, "failureThreshold": 3,
            },
        }]}}}})

    def patch_memory(self, service: str, *, limit_mi: int | None = None,
                     request_mi: int | None = None) -> None:
        resources: dict[str, Any] = {}
        if limit_mi is not None:
            resources["limits"] = {"memory": f"{limit_mi}Mi"}
        if request_mi is not None:
            resources["requests"] = {"memory": f"{request_mi}Mi"}
        self._patch_workload(service, {"spec": {"template": {"spec": {"containers": [
            {"name": service, "resources": resources}]}}}})

    def patch_configmap(self, name: str, key: str, value: str) -> None:
        self.kube.core.patch_namespaced_config_map(
            name=name, namespace=self.namespace, body={"data": {key: value}},
        )

    def patch_deployment_secret_ref(self, service: str, secret: str) -> None:
        dep = self._deployment(service)
        if dep is None:
            raise KeyError(service)
        container = dep.spec.template.spec.containers[0]
        env_from = []
        for src in container.env_from or []:
            if src.secret_ref is not None:
                env_from.append({"secretRef": {"name": secret}})
            elif src.config_map_ref is not None:
                env_from.append({"configMapRef": {"name": src.config_map_ref.name}})
        self._patch_workload(service, {"spec": {"template": {"spec": {"containers": [
            {"name": service, "envFrom": env_from}]}}}})

    def patch_service_selector(self, service: str, selector: dict[str, str]) -> None:
        self.kube.core.patch_namespaced_service(
            name=service, namespace=self.namespace,
            body={"spec": {"selector": selector}},
        )

    def patch_resource_quota(self, name: str, hard: dict[str, str]) -> None:
        self.kube.core.patch_namespaced_resource_quota(
            name=name, namespace=self.namespace, body={"spec": {"hard": hard}},
        )

    def patch_node_selector(self, service: str, selector: dict[str, str]) -> None:
        # A strategic-merge patch cannot remove map keys, so send the whole nodeSelector.
        self._patch_workload(service, {"spec": {"template": {"spec": {
            "nodeSelector": selector or None}}}})

    def delete_network_policy(self, name: str) -> None:
        self.kube.networking.delete_namespaced_network_policy(
            name=name, namespace=self.namespace,
        )

    def delete_pod(self, name: str) -> None:
        self.kube.core.delete_namespaced_pod(name=name, namespace=self.namespace)

    def rollout_restart(self, service: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._patch_workload(service, {"spec": {"template": {"metadata": {"annotations": {
            "kubectl.kubernetes.io/restartedAt": stamp}}}}})

    def change_log(self) -> list[dict[str, Any]]:
        """Deployment history for the namespace, newest last.

        The change that introduced the fault is in here, surrounded by routine entries.
        This mirrors how outages are actually diagnosed -- "what shipped just before it
        broke" -- without handing over the root-cause label.
        """
        rng = random.Random(self.seed)
        authors = ["ci-bot", "platform-oncall", "release-automation", "a.okafor", "m.tanaka"]
        routine = [
            "bumped log verbosity to info",
            "rotated TLS certificate for ingress",
            "updated dashboard annotations",
            "increased HPA cooldown to 300s",
            "tagged release candidate",
            "synced ConfigMap from git",
        ]
        entries: list[dict[str, Any]] = []
        base_rev = rng.randint(40, 90)
        for i in range(rng.randint(2, 4)):
            entries.append({
                "revision": base_rev + i,
                "author": rng.choice(authors),
                "change": rng.choice(routine),
                "age": f"{rng.randint(3, 30)}d ago",
            })
        entries.append({
            "revision": base_rev + len(entries),
            "author": rng.choice(authors),
            "change": _describe_change(self.fault.name, self.params),
            "age": f"{rng.randint(10, 90)}m ago",
        })
        return entries

    def record_action(self, tool: str, args: dict[str, Any], result: str) -> None:
        self._actions.append({"tool": tool, "args": args, "result": result})

    @property
    def actions(self) -> list[dict[str, Any]]:
        return list(self._actions)


def _describe_change(fault_name: str, p: dict[str, Any]) -> str:
    """Phrase the fault-injecting change the way a deploy log would record it.

    Deliberately descriptive rather than diagnostic: it says what was changed, never why
    it was wrong.
    """
    svc = p.get("service") or p.get("consumer") or ""
    base = fault_name.split("+")[0]
    match base:
        case "OOMKilled" | "CascadingDependencyDown":
            return f"retuned deployment/{svc} memory limit to {p.get('broken_limit_mi', '?')}Mi"
        case "ImagePullBackOff" | "RolloutStuck":
            return f"deployed {svc} image {p.get('bad_image', '?')}"
        case "BadConfigMap":
            return f"synced configmap/{p.get('configmap')} from git (key {p.get('key')} changed)"
        case "MissingSecret":
            return f"repointed deployment/{svc} envFrom to secret/{p.get('bad_secret')}"
        case "ServiceSelectorMismatch":
            return f"updated service/{svc} selector for blue-green cutover"
        case "ReadinessProbeMisconfig":
            return (f"changed deployment/{svc} readiness probe to "
                    f"{p.get('bad_path')}:{p.get('bad_port')}")
        case "ResourceQuotaExhausted":
            return f"applied resourcequota/namespace-quota (pods={p.get('quota_pods')})"
        case "NodeSelectorUnschedulable":
            return f"pinned deployment/{svc} to nodeSelector {p.get('bad_selector')}"
        case "KVCacheExhaustion":
            return f"raised {svc} max_model_len (KV cache now ~{p.get('kv_required_mi')}Mi)"
        case "NetworkPolicyDeny":
            return f"applied networkpolicy/{p.get('policy')} to {p.get('target')}"
        case "ReplicasScaledToZero":
            return f"scaled deployment/{svc} to 0 replicas"
        case "NodeNotReady":
            return f"began maintenance on node/{p.get('node')}"
        case _:
            return f"updated deployment/{svc}"


def _unschedulable(pod) -> bool:
    for cond in pod.status.conditions or []:
        if cond.type == "PodScheduled" and cond.status == "False":
            return True
    return False


__all__ = ["Episode", "TICKS_PER_STEP"]
