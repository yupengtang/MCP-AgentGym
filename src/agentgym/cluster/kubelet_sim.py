"""A deterministic kubelet replacement.

The real scheduler binds pods to nodes (so `Unschedulable`, nodeSelector and taint
behaviour are genuine Kubernetes). This module then plays the part of the kubelet: it
decides what happens to a bound pod and writes the resulting status back through the API.

The decision is a pure function of the pod spec plus the namespace's ConfigMaps and
Secrets -- never a random draw. That is what makes rewards verifiable: the same actions
against the same seed always produce the same outcome.

Simulation advances in explicit **ticks** rather than wall-clock time. The episode driver
calls `reconcile()` after each agent action, so a trajectory can be replayed exactly.

Workload properties the simulator reads from pod-template annotations:

    agentgym.io/working-set-mi   memory the process actually needs (MiB)
    agentgym.io/boot-ticks       ticks from container start to ready
    agentgym.io/config-ref       "configmap-name:KEY1,KEY2" that must exist and be valid
    agentgym.io/secret-ref       secret name that must exist
    agentgym.io/health-path      path the readiness probe must target
    agentgym.io/health-port      port the readiness probe must target
    agentgym.io/depends-on       comma-separated services that must have ready endpoints
    agentgym.io/kv-cache-mi      vLLM-style KV cache budget (MiB), used by KVCacheExhaustion
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kubernetes.client.rest import ApiException

from agentgym import config
from agentgym.cluster.client import KubeClient, mib

ANN = "agentgym.io"


class Verdict(str, Enum):
    """What the simulated kubelet decides about a container. Order matters: the first
    matching condition in `_evaluate` wins, mirroring real kubelet precedence."""

    RUNNING_READY = "RunningReady"
    NODE_LOST = "NodeLost"
    PVC_UNBOUND = "PVCUnbound"
    IMAGE_PULL_BACKOFF = "ImagePullBackOff"
    CREATE_CONFIG_ERROR = "CreateContainerConfigError"
    CONFIG_CRASHLOOP = "CrashLoopBackOff/Config"
    OOM_CRASHLOOP = "CrashLoopBackOff/OOMKilled"
    DEPENDENCY_CRASHLOOP = "CrashLoopBackOff/DependencyUnavailable"
    KV_CACHE_OOM = "CrashLoopBackOff/KVCacheExhausted"
    BOOTING = "Booting"
    PROBE_NEVER_READY = "RunningNotReady"


# Verdicts that mean "the container is not up". Used by mesh/endpoint computation.
UNREADY = frozenset(v for v in Verdict if v is not Verdict.RUNNING_READY)
CRASHING = frozenset({
    Verdict.CONFIG_CRASHLOOP, Verdict.OOM_CRASHLOOP,
    Verdict.DEPENDENCY_CRASHLOOP, Verdict.KV_CACHE_OOM,
})


@dataclass
class WorkloadSpec:
    """Declared physical behaviour of a container, read from annotations."""

    working_set_mi: float = 256.0
    boot_ticks: int = 1
    config_ref: tuple[str, tuple[str, ...]] | None = None
    secret_ref: str | None = None
    health_path: str = "/healthz"
    health_port: int = 8080
    depends_on: tuple[str, ...] = ()
    kv_cache_mi: float = 0.0

    @classmethod
    def from_annotations(cls, ann: dict[str, str] | None) -> WorkloadSpec:
        ann = ann or {}

        def get(key: str, default: str = "") -> str:
            return ann.get(f"{ANN}/{key}", default)

        config_ref = None
        if raw := get("config-ref"):
            name, _, keys = raw.partition(":")
            config_ref = (name, tuple(k for k in keys.split(",") if k))

        return cls(
            working_set_mi=float(get("working-set-mi", "256")),
            boot_ticks=int(get("boot-ticks", "1")),
            config_ref=config_ref,
            secret_ref=get("secret-ref") or None,
            health_path=get("health-path", "/healthz"),
            health_port=int(get("health-port", "8080")),
            depends_on=tuple(d for d in get("depends-on", "").split(",") if d),
            kv_cache_mi=float(get("kv-cache-mi", "0")),
        )


@dataclass
class PodSimState:
    """Per-pod bookkeeping the API server does not hold for us."""

    first_tick: int
    restarts: int = 0
    last_verdict: Verdict | None = None
    verdict_since_tick: int = 0


@dataclass
class KubeletSim:
    """Drives pod status for every namespace it is told to manage."""

    kube: KubeClient
    nodes: tuple[config.NodeSpec, ...] = config.DEFAULT_NODES
    namespaces: list[str] = field(default_factory=list)

    tick: int = 0
    _pods: dict[str, PodSimState] = field(default_factory=dict, init=False)
    _verdicts: dict[str, Verdict] = field(default_factory=dict, init=False)
    _nodes_registered: bool = field(default=False, init=False)
    _unready_nodes: set[str] = field(default_factory=set, init=False)
    # Namespaces whose control-plane dependency graph is currently unhealthy, so the
    # dependency check does not have to re-derive readiness mid-pass.
    _ready_services: dict[str, set[str]] = field(default_factory=dict, init=False)

    # --- nodes --------------------------------------------------------------

    def register_nodes(self) -> None:
        """Create Node objects so the real scheduler has somewhere to bind pods."""
        if self._nodes_registered:
            return
        for spec in self.nodes:
            capacity = {
                "cpu": str(spec.cpu),
                "memory": f"{spec.memory_gi}Gi",
                "pods": "110",
            }
            if spec.gpu:
                capacity["nvidia.com/gpu"] = str(spec.gpu)
            labels = {
                "kubernetes.io/hostname": spec.name,
                "kubernetes.io/os": "linux",
                "kubernetes.io/arch": "amd64",
                **spec.labels,
            }
            if spec.gpu_type:
                labels["nvidia.com/gpu.product"] = spec.gpu_type
            body = {
                "apiVersion": "v1",
                "kind": "Node",
                "metadata": {"name": spec.name, "labels": labels},
                "spec": {
                    "taints": [
                        {"key": k, "value": v, "effect": e} for k, v, e in spec.taints
                    ] or None,
                },
            }
            try:
                self.kube.core.create_node(body=body)
            except ApiException as exc:
                if exc.status != 409:
                    raise
            # Capacity and Ready live on the status subresource, which create ignores.
            self.kube.core.patch_node_status(
                name=spec.name,
                body={
                    "status": {
                        "capacity": capacity,
                        "allocatable": capacity,
                        "conditions": [{
                            "type": "Ready", "status": "True",
                            "reason": "KubeletReady",
                            "message": "agentgym simulated kubelet is posting ready status",
                            "lastHeartbeatTime": _now(),
                            "lastTransitionTime": _now(),
                        }],
                        "nodeInfo": {
                            "architecture": "amd64", "operatingSystem": "linux",
                            "kubeletVersion": "v1.31.4-agentgym",
                            "osImage": "AgentGym Simulated Node",
                            "containerRuntimeVersion": "agentgym://1.0",
                            "bootID": "", "kernelVersion": "5.14.0",
                            "kubeProxyVersion": "v1.31.4", "machineID": spec.name,
                            "systemUUID": spec.name,
                        },
                    }
                },
            )
            # The API server stamps node.kubernetes.io/not-ready:NoSchedule on every new
            # Node and relies on the node-lifecycle-controller to lift it once the node
            # reports Ready. We run without that controller (it would also evict our
            # simulated pods), so the simulator owns the taint itself.
            self._set_readiness_taint(spec.name, ready=True)
        self._nodes_registered = True

    _NOT_READY_TAINTS = ("node.kubernetes.io/not-ready", "node.kubernetes.io/unreachable")

    def _set_readiness_taint(self, name: str, *, ready: bool) -> None:
        node = self.kube.core.read_node(name=name)
        taints = [
            {"key": t.key, "value": t.value, "effect": t.effect}
            for t in (node.spec.taints or [])
            if t.key not in self._NOT_READY_TAINTS
        ]
        if not ready:
            taints.append({"key": "node.kubernetes.io/not-ready", "effect": "NoSchedule"})
        self.kube.core.patch_node(name=name, body={"spec": {"taints": taints}})

    def set_node_ready(self, name: str, ready: bool, reason: str = "KubeletNotReady") -> None:
        """Used by fault families that take a node out (eviction, GPU failure)."""
        self.kube.core.patch_node_status(
            name=name,
            body={"status": {"conditions": [{
                "type": "Ready", "status": "True" if ready else "False",
                "reason": "KubeletReady" if ready else reason,
                "message": "simulated", "lastHeartbeatTime": _now(),
                "lastTransitionTime": _now(),
            }]}},
        )
        self._set_readiness_taint(name, ready=ready)
        if ready:
            self._unready_nodes.discard(name)
        else:
            self._unready_nodes.add(name)

    # --- reconcile ----------------------------------------------------------

    def reconcile(self, ticks: int = 1) -> dict[str, Verdict]:
        """Advance the simulation. Returns the verdict for every managed pod."""
        self.register_nodes()
        for _ in range(max(1, ticks)):
            self.tick += 1
            for ns in self.namespaces:
                self._reconcile_namespace(ns)
        return dict(self._verdicts)

    def _bind_pvcs(self, ns: str) -> set[str]:
        """Act as the storage provisioner. Returns the names of bound claims.

        A real cluster has a CSI driver watching for Pending claims and provisioning
        volumes. Nothing here does that, so StatefulSet pods would wait forever on a
        claim that never binds -- and the distinction between "waiting for storage" and
        "asked for a StorageClass that does not exist" would collapse.
        """
        bound: set[str] = set()
        try:
            claims = self.kube.core.list_namespaced_persistent_volume_claim(ns).items
        except ApiException:
            return bound

        for pvc in claims:
            name = pvc.metadata.name
            storage_class = pvc.spec.storage_class_name or "standard"
            if storage_class not in config.KNOWN_STORAGE_CLASSES:
                continue  # no provisioner for this class: stays Pending, by design
            if pvc.status is not None and pvc.status.phase == "Bound":
                bound.add(name)
                continue
            volume = f"pv-{ns}-{name}"
            requested = ((pvc.spec.resources.requests or {}).get("storage", "1Gi")
                         if pvc.spec.resources else "1Gi")
            try:
                self.kube.core.create_persistent_volume(body={
                    "apiVersion": "v1", "kind": "PersistentVolume",
                    "metadata": {"name": volume},
                    "spec": {
                        "capacity": {"storage": requested},
                        "accessModes": list(pvc.spec.access_modes or ["ReadWriteOnce"]),
                        "persistentVolumeReclaimPolicy": "Delete",
                        "storageClassName": storage_class,
                        "hostPath": {"path": f"/tmp/agentgym/{volume}"},
                        "claimRef": {"namespace": ns, "name": name},
                    },
                })
            except ApiException as exc:
                if exc.status != 409:
                    raise
            with contextlib.suppress(ApiException):
                self.kube.core.patch_namespaced_persistent_volume_claim(
                    name=name, namespace=ns,
                    body={"spec": {"volumeName": volume}},
                )
                self.kube.core.patch_namespaced_persistent_volume_claim_status(
                    name=name, namespace=ns,
                    body={"status": {"phase": "Bound",
                                     "accessModes": list(pvc.spec.access_modes or ["ReadWriteOnce"]),
                                     "capacity": {"storage": requested}}},
                )
            bound.add(name)
        return bound

    def _reconcile_namespace(self, ns: str) -> None:
        bound_claims = self._bind_pvcs(ns)
        pods = self.kube.core.list_namespaced_pod(ns).items
        configmaps = {c.metadata.name: (c.data or {})
                      for c in self.kube.core.list_namespaced_config_map(ns).items}
        secrets = {s.metadata.name for s in self.kube.core.list_namespaced_secret(ns).items}

        # Two passes: readiness of dependencies is computed from the previous pass so a
        # dependency cycle cannot deadlock the simulation.
        ready_now = self._ready_services.setdefault(ns, set())

        fresh_ready: set[str] = set()
        for pod in pods:
            if pod.metadata.deletion_timestamp is not None:
                # Confirming termination is the kubelet's responsibility. Without this
                # the pod object lingers forever: namespaces never leave Terminating,
                # deleted pods keep holding node capacity, and ReplicaSets never create
                # replacements.
                self._reap(pod)
                continue
            key = f"{ns}/{pod.metadata.name}"
            if not pod.spec.node_name:
                # Unbound: the scheduler owns this pod's fate (Pending/Unschedulable).
                self._verdicts[key] = Verdict.BOOTING
                continue

            state = self._pods.setdefault(key, PodSimState(first_tick=self.tick))
            verdict = self._evaluate(pod, configmaps, secrets, ready_now, state,
                                     bound_claims)

            if verdict is not state.last_verdict:
                state.last_verdict = verdict
                state.verdict_since_tick = self.tick
            if verdict in CRASHING:
                # One restart per tick while crashlooping, capped so numbers stay sane.
                state.restarts = min(state.restarts + 1, 999)

            self._verdicts[key] = verdict
            if verdict is Verdict.RUNNING_READY:
                # A pod "provides" the service named by its `app` label; that is what
                # `depends-on` refers to.
                if app := _labels_of(pod).get("app"):
                    fresh_ready.add(app)
            self._write_status(pod, verdict, state)

        self._ready_services[ns] = {s for s in fresh_ready if s}

    def _evaluate(
        self,
        pod: Any,
        configmaps: dict[str, dict[str, str]],
        secrets: set[str],
        ready_services: set[str],
        state: PodSimState,
        bound_claims: set[str] | None = None,
    ) -> Verdict:
        container = pod.spec.containers[0]
        ann = (pod.metadata.annotations or {})
        work = WorkloadSpec.from_annotations(ann)

        # 0. A pod whose node stopped reporting Ready is unreachable, whatever its own
        # state. Real Kubernetes marks these Unknown and only replaces them once the pod
        # object is deleted, which is why losing a node quietly costs capacity.
        if pod.spec.node_name in self._unready_nodes:
            return Verdict.NODE_LOST

        # 0b. Storage must be attached before the container can start. A claim whose
        # StorageClass has no provisioner never binds, and the pod waits indefinitely.
        if bound_claims is not None:
            for volume in pod.spec.volumes or []:
                claim = getattr(volume, "persistent_volume_claim", None)
                if claim is not None and claim.claim_name not in bound_claims:
                    return Verdict.PVC_UNBOUND

        # 1. Image must exist in the simulated registry.
        if container.image not in config.KNOWN_IMAGES:
            return Verdict.IMAGE_PULL_BACKOFF

        # 2. Secrets must exist before the container can be created. The container's own
        # envFrom/valueFrom is authoritative here, not the annotation -- a remediation
        # that repoints the reference must take effect, exactly as it would on a real
        # kubelet, and the annotation is only a description of what the app wants.
        if _referenced_secrets(container) - secrets:
            return Verdict.CREATE_CONFIG_ERROR

        # 3. Config must exist and carry every required key.
        if work.config_ref:
            cm_name, required = work.config_ref
            data = configmaps.get(cm_name)
            if data is None:
                return Verdict.CREATE_CONFIG_ERROR
            if any(k not in data or not str(data[k]).strip() for k in required):
                return Verdict.CONFIG_CRASHLOOP

        # 4. Memory limit must cover the working set.
        limit_mi = mib(_resource(container, "limits", "memory"))
        if limit_mi and limit_mi < work.working_set_mi:
            return Verdict.OOM_CRASHLOOP

        # 5. vLLM-style KV cache must fit in whatever memory is left.
        if work.kv_cache_mi and limit_mi:
            if work.working_set_mi + work.kv_cache_mi > limit_mi:
                return Verdict.KV_CACHE_OOM

        # 6. Hard dependencies must be serving.
        for dep in work.depends_on:
            if dep not in ready_services:
                return Verdict.DEPENDENCY_CRASHLOOP

        # 7. Readiness probe has to actually point at the health endpoint.
        probe = container.readiness_probe
        if probe is not None and probe.http_get is not None:
            port = probe.http_get.port
            port = int(port) if str(port).isdigit() else _named_port(container, str(port))
            if probe.http_get.path != work.health_path or port != work.health_port:
                return Verdict.PROBE_NEVER_READY
            if probe.initial_delay_seconds is not None and probe.failure_threshold is not None:
                # A probe that gives up before the app finishes booting never succeeds.
                budget = probe.initial_delay_seconds + (
                    probe.failure_threshold * (probe.period_seconds or 10)
                )
                if budget < work.boot_ticks:
                    return Verdict.PROBE_NEVER_READY

        # 8. Otherwise it boots, then serves.
        if self.tick - state.first_tick < work.boot_ticks:
            return Verdict.BOOTING
        return Verdict.RUNNING_READY

    # --- status writing -----------------------------------------------------

    def _write_status(self, pod: Any, verdict: Verdict, state: PodSimState) -> None:
        name, ns = pod.metadata.name, pod.metadata.namespace
        container = pod.spec.containers[0]
        ready = verdict is Verdict.RUNNING_READY
        if verdict is Verdict.NODE_LOST:
            phase = "Unknown"
        elif verdict in (Verdict.RUNNING_READY, Verdict.PROBE_NEVER_READY):
            phase = "Running"
        else:
            phase = "Pending"

        if verdict is Verdict.PVC_UNBOUND:
            cstate = {"waiting": {
                "reason": "ContainerCreating",
                "message": "pod has unbound immediate PersistentVolumeClaims",
            }}
        elif verdict is Verdict.NODE_LOST:
            cstate = {"terminated": {"reason": "NodeLost", "exitCode": 255,
                                     "startedAt": _now(), "finishedAt": _now()}}
        elif verdict is Verdict.IMAGE_PULL_BACKOFF:
            cstate = {"waiting": {"reason": "ImagePullBackOff",
                                  "message": f'Back-off pulling image "{container.image}"'}}
        elif verdict is Verdict.CREATE_CONFIG_ERROR:
            cstate = {"waiting": {"reason": "CreateContainerConfigError",
                                  "message": "referenced ConfigMap/Secret not found"}}
        elif verdict is Verdict.CONFIG_CRASHLOOP:
            cstate = {"waiting": {"reason": "CrashLoopBackOff",
                                  "message": "back-off restarting failed container: invalid configuration"}}
        elif verdict is Verdict.OOM_CRASHLOOP:
            cstate = {"waiting": {"reason": "CrashLoopBackOff",
                                  "message": "back-off restarting failed container: last state OOMKilled"}}
        elif verdict is Verdict.KV_CACHE_OOM:
            cstate = {"waiting": {"reason": "CrashLoopBackOff",
                                  "message": "back-off restarting failed container: KV cache allocation exceeded memory limit"}}
        elif verdict is Verdict.DEPENDENCY_CRASHLOOP:
            cstate = {"waiting": {"reason": "CrashLoopBackOff",
                                  "message": "back-off restarting failed container: dependency unavailable"}}
        elif verdict is Verdict.BOOTING:
            cstate = {"running": {"startedAt": _now()}}
        else:  # RUNNING_READY, PROBE_NEVER_READY
            cstate = {"running": {"startedAt": _now()}}

        last_state: dict[str, Any] = {}
        if verdict is Verdict.OOM_CRASHLOOP:
            last_state = {"terminated": {
                "reason": "OOMKilled", "exitCode": 137,
                "startedAt": _now(), "finishedAt": _now(),
            }}
        elif verdict in CRASHING:
            last_state = {"terminated": {
                "reason": "Error", "exitCode": 1,
                "startedAt": _now(), "finishedAt": _now(),
            }}

        conditions = [
            {"type": "Initialized", "status": "True", "lastTransitionTime": _now()},
            {"type": "Ready", "status": "True" if ready else "False",
             "lastTransitionTime": _now(),
             **({} if ready else {"reason": "ContainersNotReady",
                                  "message": f"containers with unready status: [{container.name}]"})},
            {"type": "ContainersReady", "status": "True" if ready else "False",
             "lastTransitionTime": _now()},
            {"type": "PodScheduled", "status": "True", "lastTransitionTime": _now()},
        ]

        body = {"status": {
            "phase": phase,
            "conditions": conditions,
            "hostIP": "10.244.0.1",
            "podIP": _pod_ip(ns, name),
            "startTime": _now(),
            "containerStatuses": [{
                "name": container.name,
                "image": container.image,
                "imageID": f"agentgym://{container.image}",
                "containerID": f"agentgym://{name}",
                "ready": ready,
                "started": phase == "Running",
                "restartCount": state.restarts,
                "state": cstate,
                **({"lastState": last_state} if last_state else {}),
            }],
        }}
        try:
            self.kube.core.patch_namespaced_pod_status(name=name, namespace=ns, body=body)
        except ApiException as exc:
            if exc.status not in (404, 409):  # pod deleted or raced with a rewrite
                raise

    def _reap(self, pod: Any) -> None:
        """Complete a pod deletion the way a kubelet would: grace period zero."""
        key = f"{pod.metadata.namespace}/{pod.metadata.name}"
        self._pods.pop(key, None)
        self._verdicts.pop(key, None)
        try:
            self.kube.core.delete_namespaced_pod(
                name=pod.metadata.name, namespace=pod.metadata.namespace,
                grace_period_seconds=0,
            )
        except ApiException as exc:
            if exc.status != 404:
                raise

    def reap_namespace(self, namespace: str) -> None:
        """Force-terminate every pod in a namespace so it can finish deleting."""
        try:
            pods = self.kube.core.list_namespaced_pod(namespace).items
        except ApiException:
            return
        for pod in pods:
            self._reap(pod)

    def verdict_for(self, namespace: str, pod_name: str) -> Verdict | None:
        return self._verdicts.get(f"{namespace}/{pod_name}")

    def forget(self, namespace: str) -> None:
        """Drop per-pod state for a namespace (called on episode reset)."""
        prefix = f"{namespace}/"
        for key in [k for k in self._pods if k.startswith(prefix)]:
            del self._pods[key]
        for key in [k for k in self._verdicts if k.startswith(prefix)]:
            del self._verdicts[key]
        self._ready_services.pop(namespace, None)


# --- helpers -----------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _labels_of(pod: Any) -> dict[str, str]:
    return dict(pod.metadata.labels or {})


def _resource(container: Any, kind: str, name: str) -> str | None:
    res = container.resources
    if res is None:
        return None
    table = getattr(res, kind, None) or {}
    return table.get(name)


def _named_port(container: Any, name: str) -> int:
    for port in container.ports or []:
        if port.name == name:
            return int(port.container_port)
    return -1


def _referenced_secrets(container: Any) -> set[str]:
    """Secrets the container mounts via envFrom / valueFrom."""
    found: set[str] = set()
    for src in container.env_from or []:
        if src.secret_ref is not None:
            found.add(src.secret_ref.name)
    for env in container.env or []:
        vf = getattr(env, "value_from", None)
        if vf is not None and getattr(vf, "secret_key_ref", None) is not None:
            found.add(vf.secret_key_ref.name)
    return found


def _pod_ip(namespace: str, name: str) -> str:
    """Stable pseudo-IP so traces and endpoint lists are reproducible."""
    h = abs(hash((namespace, name)))
    return f"10.244.{(h >> 8) % 256}.{h % 254 + 1}"


__all__ = ["KubeletSim", "Verdict", "WorkloadSpec", "UNREADY", "CRASHING"]
