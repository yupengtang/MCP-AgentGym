"""Metrics, container logs, and Kubernetes Events derived from simulated cluster state.

An agent that can only read `kubectl get pods` is doing half the job. Real diagnosis
reads logs and metrics, so those have to carry the signal that identifies the fault --
and they have to be consistent with it, or the agent learns to trust the wrong evidence.

Everything is generated deterministically from (cluster state, seed).
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from agentgym.cluster.client import KubeClient, mib
from agentgym.cluster.kubelet_sim import KubeletSim, Verdict, WorkloadSpec
from agentgym.cluster.mesh import Mesh, ServiceHealth

# --- log templates -----------------------------------------------------------
# One bank per verdict. The lines are what the real components print, because that is
# what the agent must learn to recognise.

_BOOT_LINES = [
    "INFO  [main] starting {app} build={build}",
    "INFO  [config] loaded configuration from /etc/{app}/config.yaml",
    "INFO  [server] listening on 0.0.0.0:{port}",
]

_LOGS: dict[Verdict, list[str]] = {
    Verdict.RUNNING_READY: [
        "INFO  [health] readiness probe ok",
        "INFO  [server] handled {n} requests in the last 60s p50={p50}ms p95={p95}ms",
        "DEBUG [pool] active connections={conns} idle={idle}",
    ],
    Verdict.OOM_CRASHLOOP: [
        "INFO  [main] starting {app} build={build}",
        "INFO  [model] loading weights, estimated resident set {ws}MiB",
        "WARN  [mem] rss {rss}MiB approaching cgroup limit {limit}MiB",
        "ERROR [mem] allocation of {chunk}MiB failed: cannot allocate memory",
        "FATAL [runtime] out of memory: killed by cgroup oom-killer (limit {limit}MiB)",
    ],
    Verdict.KV_CACHE_OOM: [
        "INFO  [engine] initializing vLLM engine max_model_len={max_len}",
        "INFO  [engine] profiling run to determine KV cache blocks",
        "WARN  [engine] requested KV cache {kv}MiB, available after weights {avail}MiB",
        "ERROR [engine] No available memory for the cache blocks. Try increasing gpu_memory_utilization or decreasing max_model_len.",
        "FATAL [runtime] engine initialization failed",
    ],
    Verdict.CONFIG_CRASHLOOP: [
        "INFO  [main] starting {app} build={build}",
        "INFO  [config] reading ConfigMap {cm}",
        "ERROR [config] required key '{key}' is missing or empty",
        "FATAL [main] configuration validation failed, exiting with code 1",
    ],
    Verdict.CREATE_CONFIG_ERROR: [
        "ERROR [kubelet] Error: configmap or secret referenced by the pod could not be found",
    ],
    Verdict.IMAGE_PULL_BACKOFF: [
        "ERROR [kubelet] Failed to pull image \"{image}\": manifest unknown",
    ],
    Verdict.DEPENDENCY_CRASHLOOP: [
        "INFO  [main] starting {app} build={build}",
        "INFO  [deps] connecting to {dep}",
        "ERROR [deps] dial tcp: lookup {dep} failed: no endpoints available",
        "ERROR [deps] retry {i}/5 failed",
        "FATAL [main] dependency {dep} unreachable, exiting",
    ],
    Verdict.PROBE_NEVER_READY: [
        "INFO  [main] starting {app} build={build}",
        "INFO  [server] listening on 0.0.0.0:{port}",
        "WARN  [health] received probe for {probe_path} but health endpoint is {health_path}",
        "WARN  [health] returning 404 for {probe_path}",
    ],
    Verdict.NODE_LOST: [
        "WARN  [kubelet] node {node} stopped posting status",
        "ERROR [kubelet] pod status unknown: node lost contact with the control plane",
    ],
    Verdict.BOOTING: _BOOT_LINES,
}


@dataclass
class Telemetry:
    """Generates logs, metrics, and events for one namespace."""

    kube: KubeClient
    sim: KubeletSim
    mesh: Mesh
    seed: int = 0
    _events_written: set[str] = field(default_factory=set, init=False)

    # --- logs ---------------------------------------------------------------

    def pod_logs(self, namespace: str, pod: str, *, tail_lines: int = 50,
                 previous: bool = False) -> str:
        try:
            obj = self.kube.core.read_namespaced_pod(name=pod, namespace=namespace)
        except Exception:
            return f"Error from server (NotFound): pods \"{pod}\" not found"

        verdict = self.sim.verdict_for(namespace, pod) or Verdict.BOOTING
        container = obj.spec.containers[0]
        work = WorkloadSpec.from_annotations(obj.metadata.annotations)
        rng = random.Random(self._seed_for(namespace, pod))

        app = (obj.metadata.labels or {}).get("app", container.name)
        limit_mi = mib(_limit(container, "memory")) or 0
        probe = container.readiness_probe
        probe_path = probe.http_get.path if (probe and probe.http_get) else "/healthz"

        ctx = {
            "app": app,
            "build": hashlib.sha1(app.encode()).hexdigest()[:7],
            "port": work.health_port,
            "ws": int(work.working_set_mi),
            "rss": int(min(work.working_set_mi, limit_mi or work.working_set_mi) * 0.94),
            "limit": int(limit_mi),
            "chunk": rng.choice([64, 128, 256]),
            "kv": int(work.kv_cache_mi),
            "avail": int(max(0, limit_mi - work.working_set_mi)),
            "max_len": rng.choice([8192, 16384, 32768]),
            "cm": work.config_ref[0] if work.config_ref else "app-config",
            "key": work.config_ref[1][0] if (work.config_ref and work.config_ref[1]) else "MODEL_PATH",
            "image": container.image,
            "dep": work.depends_on[0] if work.depends_on else "redis",
            "n": rng.randint(800, 4000),
            "p50": rng.randint(12, 40),
            "p95": rng.randint(60, 190),
            "conns": rng.randint(4, 40),
            "idle": rng.randint(0, 8),
            "probe_path": probe_path,
            "health_path": work.health_path,
            "node": obj.spec.node_name or "unknown",
            "i": 5,
        }

        bank = _LOGS.get(verdict, _BOOT_LINES)
        # A crashlooping container replays its short life once per restart, which is
        # exactly the repetition that makes CrashLoopBackOff recognisable in real logs.
        restarts = self._restart_count(namespace, pod)
        cycles = min(restarts + 1, 6) if verdict in _LOGS and verdict != Verdict.RUNNING_READY else 1

        lines: list[str] = []
        t0 = datetime.now(UTC) - timedelta(seconds=len(bank) * cycles)
        for cycle in range(cycles):
            for i, template in enumerate(bank):
                ts = (t0 + timedelta(seconds=len(bank) * cycle + i)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                lines.append(f"{ts} {template.format(**ctx)}")
        if verdict is Verdict.RUNNING_READY:
            lines = lines * 4

        if previous and restarts == 0:
            return f"Error from server (BadRequest): previous terminated container \"{container.name}\" in pod \"{pod}\" not found"
        return "\n".join(lines[-tail_lines:])

    def _restart_count(self, namespace: str, pod: str) -> int:
        try:
            obj = self.kube.core.read_namespaced_pod(name=pod, namespace=namespace)
            statuses = obj.status.container_statuses or []
            return statuses[0].restart_count if statuses else 0
        except Exception:
            return 0

    def _seed_for(self, *parts: str) -> int:
        h = hashlib.sha256(("|".join(parts) + str(self.seed)).encode()).digest()
        return int.from_bytes(h[:8], "big")

    # --- metrics ------------------------------------------------------------

    def metrics(self, namespace: str) -> dict[str, float]:
        """Flat metric-name -> value map, Prometheus style."""
        services = self.mesh.evaluate(namespace)
        pods = self.kube.core.list_namespaced_pod(namespace).items
        out: dict[str, float] = {}

        for name, svc in services.items():
            lbl = f'{{namespace="{namespace}",service="{name}"}}'
            out[f"service_up{lbl}"] = 1.0 if svc.available else 0.0
            out[f"service_success_rate{lbl}"] = round(svc.success_rate, 4)
            out[f"service_error_rate{lbl}"] = round(svc.error_rate, 4)
            out[f"service_request_latency_p50_ms{lbl}"] = round(svc.p50_ms, 1)
            out[f"service_request_latency_p95_ms{lbl}"] = round(svc.p95_ms, 1)
            out[f"service_endpoints_ready{lbl}"] = float(svc.ready_endpoints)
            out[f"service_endpoints_total{lbl}"] = float(len(svc.endpoints))

        for pod in pods:
            pname = pod.metadata.name
            verdict = self.sim.verdict_for(namespace, pname)
            work = WorkloadSpec.from_annotations(pod.metadata.annotations)
            container = pod.spec.containers[0]
            limit_mi = mib(_limit(container, "memory")) or 0.0
            lbl = f'{{namespace="{namespace}",pod="{pname}"}}'
            statuses = pod.status.container_statuses or []
            restarts = statuses[0].restart_count if statuses else 0

            # A container that dies on allocation sits pinned at its ceiling.
            if verdict in (Verdict.OOM_CRASHLOOP, Verdict.KV_CACHE_OOM):
                usage = limit_mi
            elif verdict is Verdict.RUNNING_READY:
                usage = min(work.working_set_mi, limit_mi or work.working_set_mi)
            else:
                usage = 0.0
            out[f"container_memory_working_set_mib{lbl}"] = round(usage, 1)
            out[f"container_memory_limit_mib{lbl}"] = round(limit_mi, 1)
            out[f"container_restarts_total{lbl}"] = float(restarts)
            out[f"container_ready{lbl}"] = 1.0 if verdict is Verdict.RUNNING_READY else 0.0
            if work.kv_cache_mi:
                headroom = max(0.0, limit_mi - work.working_set_mi)
                util = 1.0 if headroom <= 0 else min(1.0, work.kv_cache_mi / headroom)
                out[f"vllm_kv_cache_utilization{lbl}"] = round(util, 3)
                out[f"vllm_kv_cache_required_mib{lbl}"] = round(work.kv_cache_mi, 1)
                out[f"vllm_kv_cache_available_mib{lbl}"] = round(headroom, 1)
        return out

    def query(self, namespace: str, expr: str) -> list[dict]:
        """PromQL-lite: `metric_name` or `metric_name{label="value",...}`.

        Supports exact-match label filters only. That is enough for diagnosis and keeps
        the tool surface honest -- an agent cannot bluff its way through with a rate()
        expression the environment would silently ignore.
        """
        all_metrics = self.metrics(namespace)
        match = re.match(r"^\s*([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(\{(.*)\})?\s*$", expr)
        if not match:
            raise ValueError(
                f"unsupported expression {expr!r}; use metric_name or metric_name{{label=\"value\"}}"
            )
        wanted_name = match.group(1)
        filters: dict[str, str] = {}
        if match.group(3):
            for part in match.group(3).split(","):
                if not part.strip():
                    continue
                k, _, v = part.partition("=")
                filters[k.strip()] = v.strip().strip('"')

        results = []
        for key, value in sorted(all_metrics.items()):
            name, _, labelblob = key.partition("{")
            if name != wanted_name:
                continue
            labels = dict(re.findall(r'(\w+)="([^"]*)"', "{" + labelblob))
            if all(labels.get(k) == v for k, v in filters.items()):
                results.append({"metric": name, "labels": labels, "value": value})
        return results

    # --- events -------------------------------------------------------------

    def emit_events(self, namespace: str) -> None:
        """Write Kubernetes Events matching current pod verdicts, once per transition."""
        for pod in self.kube.core.list_namespaced_pod(namespace).items:
            pname = pod.metadata.name
            verdict = self.sim.verdict_for(namespace, pname)
            if verdict is None or verdict is Verdict.RUNNING_READY:
                continue
            key = f"{namespace}/{pname}/{verdict.value}"
            if key in self._events_written:
                continue
            self._events_written.add(key)
            reason, message, etype = _event_for(verdict, pod)
            stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            body = {
                "apiVersion": "v1", "kind": "Event",
                "metadata": {"generateName": f"{pname}.", "namespace": namespace},
                "involvedObject": {
                    "kind": "Pod", "namespace": namespace, "name": pname,
                    "uid": pod.metadata.uid, "apiVersion": "v1",
                },
                "reason": reason, "message": message, "type": etype,
                "source": {"component": "kubelet", "host": pod.spec.node_name or ""},
                "firstTimestamp": stamp, "lastTimestamp": stamp, "count": 1,
            }
            try:
                self.kube.core.create_namespaced_event(namespace=namespace, body=body)
            except Exception:
                pass  # events are advisory; never fail an episode over one


def _event_for(verdict: Verdict, pod) -> tuple[str, str, str]:
    image = pod.spec.containers[0].image
    match verdict:
        case Verdict.IMAGE_PULL_BACKOFF:
            return ("Failed", f'Failed to pull image "{image}": manifest unknown', "Warning")
        case Verdict.OOM_CRASHLOOP:
            return ("BackOff", "Back-off restarting failed container: last state OOMKilled", "Warning")
        case Verdict.KV_CACHE_OOM:
            return ("BackOff", "Back-off restarting failed container: KV cache allocation failed", "Warning")
        case Verdict.CONFIG_CRASHLOOP:
            return ("BackOff", "Back-off restarting failed container: invalid configuration", "Warning")
        case Verdict.CREATE_CONFIG_ERROR:
            return ("Failed", "Error: configmap or secret referenced by the pod could not be found", "Warning")
        case Verdict.DEPENDENCY_CRASHLOOP:
            return ("BackOff", "Back-off restarting failed container: dependency unreachable", "Warning")
        case Verdict.PROBE_NEVER_READY:
            return ("Unhealthy", "Readiness probe failed: HTTP probe returned statuscode 404", "Warning")
        case Verdict.NODE_LOST:
            return ("NodeNotReady", "Node stopped posting status; pod state is Unknown", "Warning")
        case _:
            return ("Scheduled", "Successfully assigned pod to node", "Normal")


def _limit(container, name: str) -> str | None:
    res = container.resources
    if res is None or not res.limits:
        return None
    return res.limits.get(name)


__all__ = ["Telemetry", "ServiceHealth"]
